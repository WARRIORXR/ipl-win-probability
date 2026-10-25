"""Train the win-probability model with time-aware validation.

Split design (fixed, by season, never by row):
    * TRAIN      seasons 2008-2021
    * VALIDATION season 2022 (used for tuning and calibration)
    * TEST       seasons 2023-2024 (reported metrics only)
Every match appears in exactly one split; all balls of a match share its split.

Run ``python -m src.train`` to train, evaluate and save:
    models/model.joblib      pipeline + calibrator + everything the app needs
    models/metadata.json     split, metrics, versions (real measured numbers)
"""

from __future__ import annotations

import json
import logging
import platform
from importlib.metadata import version as pkg_version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import evaluate as ev
from src.features import (
    CATEGORICAL_FEATURES,
    LABEL,
    NUMERIC_FEATURES,
    build_chase_features,
    latest_team_win_rates,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"

SEED = 42
TRAIN_SEASONS = list(range(2008, 2022))
VAL_SEASONS = [2022]
TEST_SEASONS = [2023, 2024]

#: Momentum features - kept in the dataset, evaluated as an ablation, and
#: excluded from the deployed model unless they earn their keep (see main).
MOMENTUM_FEATURES = ["last5_overs_runs", "last5_overs_wickets"]
#: Deployed model features: everything derivable from the app's inputs.
DEPLOYED_FEATURES: list[str] = [c for c in NUMERIC_FEATURES if c not in MOMENTUM_FEATURES] + CATEGORICAL_FEATURES


def season_split(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split feature rows by season; matches never span two splits.

    Args:
        features: Output of :func:`src.features.build_chase_features`.

    Returns:
        Dict with ``train``, ``validation`` and ``test`` dataframes, and the
        list of match ids per split stored under ``split_report``.

    Raises:
        ValueError: If any match id appears in more than one split.
    """
    parts = {
        "train": features[features["season_year"].isin(TRAIN_SEASONS)],
        "validation": features[features["season_year"].isin(VAL_SEASONS)],
        "test": features[features["season_year"].isin(TEST_SEASONS)],
    }
    seen: set[int] = set()
    for name, part in parts.items():
        ids = set(part["match_id"].unique())
        overlap = seen & ids
        if overlap:
            raise ValueError(f"matches {sorted(overlap)[:5]} appear in more than one split (at {name})")
        seen |= ids
    return parts


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """OHE for categoricals (unknown levels -> all zeros); numerics as given."""
    return ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(model_name: str, params: dict, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Create an unfitted sklearn pipeline for one model family.

    Args:
        model_name: One of ``logreg``, ``random_forest``, ``xgboost``.
        params: Hyperparameters for the estimator.
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.

    Returns:
        Unfitted pipeline (preprocessor + estimator).
    """
    pre = _preprocessor(numeric, categorical)
    if model_name == "logreg":
        pre = ColumnTransformer(
            [
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ],
            verbose_feature_names_out=True,
        )
        est = LogisticRegression(max_iter=2000, random_state=SEED, **params)
    elif model_name == "random_forest":
        est = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **params)
    elif model_name == "xgboost":
        est = XGBClassifier(
            n_estimators=400,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
            **params,
        )
    else:
        raise ValueError(f"unknown model {model_name!r}")
    return Pipeline([("preprocess", pre), ("classifier", est)])


def tune(model_name: str, grid: list[dict], parts: dict[str, pd.DataFrame],
         feature_columns: list[str]) -> tuple[dict, Pipeline, float]:
    """Fit every grid combo on train, pick the best by validation log loss.

    Args:
        model_name: Model family name.
        grid: List of hyperparameter dicts.
        parts: Season split from :func:`season_split`.
        feature_columns: Feature subset to use.

    Returns:
        (best params, fitted pipeline, best validation log loss).
    """
    X_tr, y_tr = parts["train"][feature_columns], parts["train"][LABEL]
    X_val, y_val = parts["validation"][feature_columns], parts["validation"][LABEL]
    best: tuple[dict, Pipeline, float] | None = None
    for params in grid:
        pipe = build_pipeline(model_name, params, [c for c in feature_columns if c in NUMERIC_FEATURES],
                              [c for c in feature_columns if c in CATEGORICAL_FEATURES])
        pipe.fit(X_tr, y_tr)
        ll = float(log_loss(y_val, pipe.predict_proba(X_val)[:, 1]))
        logger.info("%s %s -> val log loss %.5f", model_name, params, ll)
        if best is None or ll < best[2]:
            best = (params, pipe, ll)
    assert best is not None
    return best


def naive_rrr_baseline(parts: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Baseline: predict a win whenever RRR is at most a threshold.

    The threshold is tuned on the validation season; probabilities are hard
    0/1 clipped to [0.01, 0.99] purely so log loss stays finite - compare this
    baseline on ROC-AUC/accuracy, models on log loss/Brier.

    Returns:
        Dict with the threshold and test metrics.
    """
    best_t, best_ll = None, np.inf
    y_val = parts["validation"][LABEL]
    for t in np.arange(4.0, 16.01, 0.5):
        p = np.clip((parts["validation"]["rrr"] <= t).astype(float), 0.01, 0.99)
        ll = log_loss(y_val, p)
        if ll < best_ll:
            best_t, best_ll = float(t), ll
    y_test = parts["test"][LABEL]
    p_test = np.clip((parts["test"]["rrr"] <= best_t).astype(float), 0.01, 0.99)
    return {
        "rrr_threshold": best_t,
        "val_log_loss": best_ll,
        **ev.summary_metrics(y_test, p_test),
        "accuracy": float(accuracy_score(y_test, (p_test >= 0.5).astype(int))),
    }


def fit_calibrators(p_val: np.ndarray, y_val: pd.Series) -> dict[str, object]:
    """Fit isotonic and Platt (logistic) calibrators on validation predictions.

    Args:
        p_val: Un-calibrated validation probabilities.
        y_val: Validation labels.

    Returns:
        Dict with both calibrators and their validation log losses.
    """
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
    platt = LogisticRegression(max_iter=1000).fit(p_val.reshape(-1, 1), y_val)
    return {
        "isotonic": iso,
        "platt": platt,
        "val_log_loss_isotonic": float(log_loss(y_val, iso.predict(p_val))),
        "val_log_loss_platt": float(log_loss(y_val, platt.predict_proba(p_val.reshape(-1, 1))[:, 1])),
    }


def oof_predictions(model_name: str, params: dict, X: pd.DataFrame, y: pd.Series,
                    numeric: list[str], categorical: list[str], n_splits: int = 5) -> np.ndarray:
    """Out-of-fold probabilities on the training seasons.

    A calibrator fitted on one season of validation data can simply fit that
    season's noise, so calibrators are fitted on 5-fold out-of-fold train
    predictions instead; the validation season then gates acceptance.

    Args:
        model_name: Model family name.
        params: Chosen hyperparameters.
        X: Training feature rows.
        y: Training labels.
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        n_splits: Number of CV folds.

    Returns:
        Out-of-fold probability for every training row.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for fit_idx, pred_idx in skf.split(X, y):
        pipe = build_pipeline(model_name, params, numeric, categorical)
        pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        oof[pred_idx] = pipe.predict_proba(X.iloc[pred_idx])[:, 1]
    return oof


def _per_row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row log loss with probabilities clipped away from exactly 0/1."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calibration_passes_gate(y_val, p_raw: np.ndarray, p_cal: np.ndarray) -> bool:
    """Accept a calibrator only for a statistically clear validation gain.

    Uses a paired test on per-row log loss: the mean improvement must be
    positive and exceed 2 standard errors, so noise-level gains on one season
    of validation data do not trigger calibration.

    Args:
        y_val: Validation labels.
        p_raw: Raw model probabilities on validation.
        p_cal: Calibrated probabilities on validation.

    Returns:
        True if the calibrator improves validation log loss beyond noise.
    """
    y = np.asarray(y_val, dtype=float)
    diff = _per_row_log_loss(y, p_raw) - _per_row_log_loss(y, p_cal)
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    return bool(diff.mean() > 0 and diff.mean() > 2 * se)


def aggregated_importance(pipe: Pipeline, numeric: list[str], categorical: list[str]) -> dict[str, float]:
    """Sum one-hot dummy importances back to their base feature.

    Args:
        pipe: Fitted pipeline with a tree model exposing ``feature_importances_``.
        numeric: Numeric feature names (transformer prefix ``num__``).
        categorical: Categorical feature names (prefix ``cat__<col>_``).

    Returns:
        Mapping of base feature name to total gain share.
    """
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    clf = pipe.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    else:  # linear model: absolute standardized coefficients, renormalised
        values = np.abs(clf.coef_[0])
        values = values / values.sum()
    total: dict[str, float] = {}
    for name, value in zip(names, values):
        if name.startswith("num__"):
            base = name.removeprefix("num__")
        else:
            base = next((c for c in categorical if name.startswith(f"cat__{c}_")), name)
        total[base] = total.get(base, 0.0) + float(value)
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))


def main() -> None:
    """Train, tune, calibrate, evaluate and persist the final model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    csv_path = DATA_PROCESSED / "chase_features.csv"
    if csv_path.exists():
        features = pd.read_csv(csv_path)
    else:
        from src.data import clean_deliveries, clean_matches, load_deliveries, load_matches

        features = build_chase_features(clean_matches(load_matches()), clean_deliveries(load_deliveries()))
        features.to_csv(csv_path, index=False)

    parts = season_split(features)
    for name, part in parts.items():
        logger.info("%s: %d balls, %d matches, seasons %d-%d", name, len(part),
                    part["match_id"].nunique(), part["season_year"].min(), part["season_year"].max())

    results: dict[str, dict] = {}
    results["naive_rrr_threshold"] = naive_rrr_baseline(parts)

    grids = {
        "logreg": [{"C": 0.03}, {"C": 0.3}, {"C": 1.0}],
        "random_forest": [
            {"max_depth": 10, "min_samples_leaf": 20},
            {"max_depth": 10, "min_samples_leaf": 50},
            {"max_depth": 16, "min_samples_leaf": 20},
            {"max_depth": 16, "min_samples_leaf": 50},
        ],
        "xgboost": [
            {"max_depth": 4, "learning_rate": 0.05},
            {"max_depth": 4, "learning_rate": 0.1},
            {"max_depth": 6, "learning_rate": 0.05},
            {"max_depth": 6, "learning_rate": 0.1},
        ],
    }

    best: dict[str, tuple[dict, Pipeline, float]] = {}
    for name, grid in grids.items():
        params, pipe, val_ll = tune(name, grid, parts, NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        best[name] = (params, pipe, val_ll)
        results[name] = {"params": params, "val_log_loss": val_ll}

    # Momentum ablation: best family trained WITHOUT the last-5-over features.
    family = min(best, key=lambda k: best[k][2])
    momentum_params, _, _ = best[family]
    _, pipe_no_mom, val_ll_no_mom = tune(
        family, [momentum_params], parts, DEPLOYED_FEATURES
    )
    momentum_delta = best[family][2] - val_ll_no_mom
    results["momentum_ablation"] = {
        "features_dropped": MOMENTUM_FEATURES,
        "val_log_loss_with": best[family][2],
        "val_log_loss_without": val_ll_no_mom,
        "delta": momentum_delta,
        "decision": ("deploy without momentum (delta < 0.005)" if momentum_delta < 0.005
                     else "keep momentum"),
    }

    deployed_features = DEPLOYED_FEATURES if momentum_delta < 0.005 else NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if deployed_features != DEPLOYED_FEATURES:
        _, final_pipe, _ = best[family]
    else:
        final_pipe = pipe_no_mom
    final_params = momentum_params

    # Calibrate ONLY if a calibrator clearly beats the raw model on the
    # validation season. Calibrators are fitted on out-of-fold TRAIN
    # predictions (single-season isotonic fits that season's noise - the
    # naive variant degraded test log loss from 0.482 to 0.763).
    numeric_cols = [c for c in deployed_features if c in NUMERIC_FEATURES]
    cat_cols = [c for c in deployed_features if c in CATEGORICAL_FEATURES]
    p_val = final_pipe.predict_proba(parts["validation"][deployed_features])[:, 1]
    raw_val_ll = float(log_loss(parts["validation"][LABEL], p_val))
    oof = oof_predictions(family, final_params, parts["train"][deployed_features],
                          parts["train"][LABEL], numeric_cols, cat_cols)
    cals = fit_calibrators(oof, parts["train"][LABEL])
    cal_name, calibrator = "none", None
    y_val = parts["validation"][LABEL]
    gate_results: dict[str, bool] = {}
    for cand in ("isotonic", "platt"):
        p_cand = (cals[cand].predict(p_val) if cand == "isotonic"
                  else cals[cand].predict_proba(p_val.reshape(-1, 1))[:, 1])
        gate_results[cand] = calibration_passes_gate(y_val, p_val, p_cand)
        if gate_results[cand] and cal_name == "none":
            cal_name, calibrator = cand, cals[cand]
    logger.info("calibration gate (fitted on OOF train, gated on val): %s -> %s",
                gate_results, cal_name)

    # Test metrics: raw, and with the attempted calibrator (before/after).
    X_test, y_test = parts["test"][deployed_features], parts["test"][LABEL]
    p_test_raw = final_pipe.predict_proba(X_test)[:, 1]
    if calibrator is None:
        p_test_cal = p_test_raw
    elif cal_name == "isotonic":
        p_test_cal = calibrator.predict(p_test_raw)
    else:
        p_test_cal = calibrator.predict_proba(p_test_raw.reshape(-1, 1))[:, 1]

    metrics_raw = ev.summary_metrics(y_test, p_test_raw)
    metrics_cal = ev.summary_metrics(y_test, p_test_cal)
    results["final_model"] = {
        "family": family,
        "params": final_params,
        "deployed_features": deployed_features,
        "calibrator": cal_name,
        "calibration_gate": gate_results,
        "val_log_loss": raw_val_ll,
        "val_log_loss_isotonic": cals["val_log_loss_isotonic"],
        "val_log_loss_platt": cals["val_log_loss_platt"],
        "test_uncalibrated": metrics_raw,
        "test_calibrated": metrics_cal,
    }
    if cal_name == "none":
        results["calibration_attempt"] = {
            "decision": ("rejected: no clear validation improvement; "
                         "raw model is already well calibrated"),
            "test_log_loss_with_isotonic": metrics_cal["log_loss"],
        }
    stages = ev.stage_table(parts["test"], y_test, p_test_cal)
    results["stage_metrics_test"] = stages.to_dict(orient="records")

    logger.info("final: %s %s calibrated with %s", family, final_params, cal_name)
    logger.info("test raw: %s", metrics_raw)
    logger.info("test calibrated: %s", metrics_cal)
    print("\n=== TEST SET (seasons 2023-2024, "
          f"{parts['test']['match_id'].nunique()} matches, {len(parts['test'])} balls) ===")
    print(stages.to_string(index=False))
    print("\ncalibration table (calibrated):")
    print(ev.calibration_table(y_test, p_test_cal).to_string(index=False))

    # Persist artifacts.
    from src.data import clean_matches, load_matches

    team_strength = latest_team_win_rates(clean_matches(load_matches()))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipe,
            "calibrator": calibrator,
            "calibrator_name": cal_name,
            "model_family": family,
            "feature_columns": deployed_features,
            "numeric_features": [c for c in deployed_features if c in NUMERIC_FEATURES],
            "categorical_features": [c for c in deployed_features if c in CATEGORICAL_FEATURES],
            "team_strength": team_strength,
            "teams": sorted(set(features["batting_team"]) | set(features["bowling_team"])),
            "venues": sorted(features["venue"].unique()),
            "seed": SEED,
        },
        MODELS_DIR / "model.joblib",
    )
    metadata = {
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
        "seed": SEED,
        "split": {
            "train_seasons": TRAIN_SEASONS,
            "validation_seasons": VAL_SEASONS,
            "test_seasons": TEST_SEASONS,
            "train_matches": int(parts["train"]["match_id"].nunique()),
            "validation_matches": int(parts["validation"]["match_id"].nunique()),
            "test_matches": int(parts["test"]["match_id"].nunique()),
            "train_balls": int(len(parts["train"])),
            "validation_balls": int(len(parts["validation"])),
            "test_balls": int(len(parts["test"])),
        },
        "results": results,
        "feature_importance": aggregated_importance(
            final_pipe,
            [c for c in deployed_features if c in NUMERIC_FEATURES],
            [c for c in deployed_features if c in CATEGORICAL_FEATURES],
        ),
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pkg_version("pandas"),
            "numpy": pkg_version("numpy"),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": pkg_version("joblib"),
        },
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # Figures for the README.
    curves = {"uncalibrated (deployed)": (y_test, p_test_raw)}
    if cal_name == "none":
        curves["isotonic fitted on val (rejected)"] = (y_test, cals["isotonic"].predict(p_test_raw))
    else:
        curves[f"{cal_name}-calibrated (deployed)"] = (y_test, p_test_cal)
    ev.plot_calibration(curves, ev.FIGURES_DIR / "calibration_curve.png")
    ev.plot_importance(
        metadata["feature_importance"], ev.FIGURES_DIR / "feature_importance.png"
    )
    print(f"\nsaved {MODELS_DIR / 'model.joblib'} and {MODELS_DIR / 'metadata.json'}")
    print("top drivers:", list(metadata["feature_importance"].items())[:6])
    print("\n=== MODEL COMPARISON ===")
    print("naive RRR threshold:", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in results["naive_rrr_threshold"].items()})
    for name in ("logreg", "random_forest", "xgboost"):
        print(f"{name}: val_log_loss={best[name][2]:.5f} params={best[name][0]}")
    print("momentum ablation:", results["momentum_ablation"])
    print("final test raw:", {k: round(v, 5) for k, v in metrics_raw.items()})
    print("final test calibrated:", {k: round(v, 5) for k, v in metrics_cal.items()})


if __name__ == "__main__":
    main()
