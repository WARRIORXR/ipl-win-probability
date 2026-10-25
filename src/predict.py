"""Load trained artifacts and predict a single match situation.

Used by the Streamlit app and tests. The deployed model is the pipeline saved
by ``src/train.py``; situation inputs are the seven values the app collects.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.features import MAX_RUN_RATE, TOTAL_WICKETS

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict:
    """Load the saved model artifacts (cached so the app loads once).

    Args:
        path: Optional override of the joblib path (used by tests).

    Returns:
        Dict with ``pipeline``, ``calibrator``, feature lists, team strength,
        teams and venues.
    """
    import joblib

    return joblib.load(Path(path) if path else MODEL_PATH)


def predict_situation(
    batting_team: str,
    bowling_team: str,
    venue: str,
    target: int,
    score: int,
    overs_completed: int,
    balls_completed: int,
    wickets_lost: int,
    balls_total: int = 120,
    model_path: str | None = None,
) -> dict:
    """Predict the batting team's win probability for one match situation.

    Momentum features (runs/wickets in the last 5 overs) are not part of the
    app's inputs; the deployed model does not use them (see the momentum
    ablation in ``models/metadata.json``), and any row fields it ignores are
    filled with a steady-rate estimate purely for transparency.

    Args:
        batting_team: Team batting (chasing).
        bowling_team: Team bowling (defending).
        venue: Canonical venue name.
        target: Target for the chasing team.
        score: Current score of the chasing team.
        overs_completed: Full overs completed (0-19).
        balls_completed: Additional balls in the current over (0-5).
        wickets_lost: Wickets lost so far (0-10).
        balls_total: Total deliveries in the chase (default 120).
        model_path: Optional artifact override.

    Returns:
        Dict with ``p_bat`` (chasing team), ``p_bowl``, the derived situation
        stats (crr, rrr, runs_left, balls_left) and the feature row used.
    """
    art = load_model(model_path)
    balls_bowled = overs_completed * 6 + balls_completed
    runs_left = max(target - score, 0)
    balls_left = max(balls_total - balls_bowled, 0)
    crr = min(score / (balls_bowled / 6), MAX_RUN_RATE) if balls_bowled > 0 else 0.0
    rrr = min(runs_left / (balls_left / 6), MAX_RUN_RATE) if balls_left > 0 else (
        MAX_RUN_RATE if runs_left > 0 else 0.0)
    strength = art["team_strength"]
    row = {
        "target": int(target),
        "runs_so_far": int(score),
        "runs_left": runs_left,
        "balls_left": balls_left,
        "balls_bowled": balls_bowled,
        "wickets_out": int(wickets_lost),
        "wickets_left": TOTAL_WICKETS - int(wickets_lost),
        "crr": crr,
        "rrr": rrr,
        # momentum: not app inputs; steady-rate estimate, unused by the model
        "last5_overs_runs": round(min(crr * 5.0, score)),
        "last5_overs_wickets": round(wickets_lost * min(balls_bowled, 30) / balls_bowled) if balls_bowled else 0,
        "rolling_win_rate_bat": strength.get(batting_team, 0.5),
        "rolling_win_rate_bowl": strength.get(bowling_team, 0.5),
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
    }
    X = pd.DataFrame([row])[art["feature_columns"]]
    p_raw = float(art["pipeline"].predict_proba(X)[:, 1][0])
    cal = art["calibrator"]
    if cal is not None:
        p = float(cal.predict([p_raw])[0]) if art["calibrator_name"] == "isotonic" else float(
            cal.predict_proba([[p_raw]])[:, 1][0])
    else:
        p = p_raw
    return {
        "p_bat": p,
        "p_bowl": 1.0 - p,
        "crr": crr,
        "rrr": rrr,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": TOTAL_WICKETS - int(wickets_lost),
        "features": row,
    }
