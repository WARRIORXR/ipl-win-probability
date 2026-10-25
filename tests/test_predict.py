"""Tests for src/predict.py: artifact loading and situation prediction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.predict import MODEL_PATH, load_model, predict_situation

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="models/model.joblib not trained yet (run python -m src.train)"
)


@pytest.fixture(scope="module")
def art() -> dict:
    """Loaded model artifacts for the module."""
    return load_model()


def test_artifact_contains_app_needs(art: dict) -> None:
    """The app needs the pipeline, teams, venues and team strength."""
    for key in ("pipeline", "teams", "venues", "team_strength", "feature_columns"):
        assert key in art
    assert len(art["teams"]) >= 10
    assert len(art["venues"]) >= 30


def test_deployed_features_match_metadata(art: dict) -> None:
    """Saved feature list must equal the metadata's deployed list."""
    meta = json.loads((MODEL_PATH.parent / "metadata.json").read_text())
    assert art["feature_columns"] == meta["results"]["final_model"]["deployed_features"]
    assert "last5_overs_runs" not in art["feature_columns"], (
        "momentum features must not be deployed (not derivable from app inputs)"
    )


def test_comfortable_chase_beats_difficult_chase(art: dict) -> None:
    """Same target and over: 150/2 is far ahead of 90/2."""
    easy = predict_situation("Mumbai Indians", "Punjab Kings", art["venues"][0],
                             180, 150, 15, 0, 2)
    hard = predict_situation("Mumbai Indians", "Punjab Kings", art["venues"][0],
                             180, 90, 15, 0, 2)
    assert 0.0 <= easy["p_bat"] <= 1.0
    assert 0.0 <= hard["p_bat"] <= 1.0
    assert easy["p_bat"] > hard["p_bat"] + 0.3


def test_wickets_lower_win_probability() -> None:
    """150/7 must rate below 150/2 in the identical situation."""
    settled = predict_situation("Chennai Super Kings", "Rajasthan Royals",
                                "MA Chidambaram Stadium, Chepauk", 180, 150, 15, 0, 2)
    wobbling = predict_situation("Chennai Super Kings", "Rajasthan Royals",
                                 "MA Chidambaram Stadium, Chepauk", 180, 150, 15, 0, 7)
    assert settled["p_bat"] > wobbling["p_bat"]
    assert wobbling["wickets_left"] == 3


def test_balls_left_and_rrr_derived_correctly() -> None:
    """12.3 overs = 75 balls; RRR and balls left must follow the convention."""
    res = predict_situation("Kolkata Knight Riders", "Sunrisers Hyderabad",
                            "Eden Gardens, Kolkata", 200, 100, 12, 3, 4)
    assert res["features"]["balls_bowled"] == 75
    assert res["balls_left"] == 45
    assert res["runs_left"] == 100
    assert res["rrr"] == pytest.approx(100 / (45 / 6))


def test_unknown_team_and_venue_do_not_crash() -> None:
    """One-hot encoder must tolerate unseen labels (handle_unknown=ignore)."""
    res = predict_situation("Mars Martians", "Venus Volcanoes", "Nowhere Park", 150, 40, 10, 0, 3)
    assert 0.0 <= res["p_bat"] <= 1.0
