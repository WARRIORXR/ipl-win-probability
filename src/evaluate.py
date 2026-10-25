"""Evaluation helpers: metrics, calibration tables and plots.

All figures are written to ``reports/figures/`` with the Matplotlib Agg
backend so they can be produced headless (scripts, notebooks, Streamlit).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"

#: Chase stages by legal balls bowled: (label, min inclusive, max inclusive).
STAGES: list[tuple[str, int, int]] = [
    ("overs 1-6", 1, 36),
    ("overs 7-15", 37, 90),
    ("overs 16-20", 91, 120),
]


def summary_metrics(y_true, p_pred) -> dict[str, float]:
    """Log loss, Brier score and ROC-AUC for predicted probabilities.

    Args:
        y_true: Binary labels.
        p_pred: Predicted probability of the positive class (chase won).

    Returns:
        Dict with ``log_loss``, ``brier`` and ``roc_auc``.
    """
    return {
        "log_loss": float(log_loss(y_true, p_pred)),
        "brier": float(brier_score_loss(y_true, p_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_pred)),
    }


def calibration_table(y_true, p_pred, n_bins: int = 10) -> pd.DataFrame:
    """Grouped calibration: mean predicted probability vs observed win rate.

    Args:
        y_true: Binary labels.
        p_pred: Predicted probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        Dataframe with bin, count, mean predicted probability, observed rate.
    """
    df = pd.DataFrame({"p": np.asarray(p_pred, dtype=float), "y": np.asarray(y_true, dtype=int)})
    df["bin"] = np.minimum((df["p"] * n_bins).astype(int), n_bins - 1)
    grouped = df.groupby("bin").agg(
        n=("y", "size"),
        mean_predicted=("p", "mean"),
        observed_win_rate=("y", "mean"),
    )
    return grouped.reset_index()


def stage_table(features: pd.DataFrame, y_true, p_pred) -> pd.DataFrame:
    """Metrics by stage of the chase (overs 1-6, 7-15, 16-20).

    Args:
        features: Feature dataframe (needs the ``balls_bowled`` column).
        y_true: Binary labels aligned with ``features``.
        p_pred: Predicted probabilities aligned with ``features``.

    Returns:
        One row per stage with n, log loss, Brier, ROC-AUC.
    """
    df = pd.DataFrame(
        {"balls": features["balls_bowled"].to_numpy(), "y": np.asarray(y_true), "p": np.asarray(p_pred)}
    )
    rows = []
    for label, lo, hi in STAGES:
        part = df[(df["balls"] >= lo) & (df["balls"] <= hi)]
        metrics = summary_metrics(part["y"], part["p"]) if len(part) else {}
        rows.append({"stage": label, "n_balls": len(part), **metrics})
    return pd.DataFrame(rows)


def plot_calibration(curves: dict[str, tuple], path: Path) -> Path:
    """Plot reliability curves (one line per series) and save to ``path``.

    Args:
        curves: Mapping of series name to (y_true, p_pred).
        path: Output PNG path.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for name, (y, p) in curves.items():
        cal = calibration_table(y, p)
        ax.plot(cal["mean_predicted"], cal["observed_win_rate"], marker="o", markersize=3.5, label=name)
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("observed win rate")
    ax.set_title("Calibration (test seasons)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_importance(importance: dict[str, float], path: Path, top: int = 15) -> Path:
    """Horizontal bar chart of aggregated feature importances.

    Args:
        importance: Mapping of base feature name to importance.
        path: Output PNG path.
        top: How many features to show.

    Returns:
        The path written.
    """
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    ax.barh([k for k, _ in items], [v for _, v in items], color="#1f77b4")
    ax.set_xlabel("importance (gain share)")
    ax.set_title("What drives the prediction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_win_curve(balls_bowled: list[int], probs: list[float], wickets_balls: list[int],
                   path: Path | None = None, title: str = "Win probability over the chase",
                   color: str = "#1F77B4"):
    """Plot a single chase's win-probability curve with wicket markers.

    Args:
        balls_bowled: Legal balls bowled at each point.
        probs: Win probability of the chasing team at each point.
        wickets_balls: ``balls_bowled`` values where a wicket fell (markers).
        path: Optional output PNG path; None keeps the figure open for the
            caller (e.g. ``st.pyplot``).
        title: Chart title.
        color: Line/fill colour (the app passes the batting team's colour).

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150)
    ax.plot(balls_bowled, probs, linewidth=1.8, color=color)
    ax.fill_between(balls_bowled, probs, color=color, alpha=0.14)
    if wickets_balls:
        wk_probs = [probs[balls_bowled.index(b)] for b in wickets_balls]
        ax.scatter(wickets_balls, wk_probs, marker="x", color="crimson", zorder=3, label="wicket")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("legal balls bowled")
    ax.set_ylabel("P(chasing team wins)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return fig
