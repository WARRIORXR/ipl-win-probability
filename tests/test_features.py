"""Tests for src/features.py using small synthetic matches.

These tests do NOT need the real data files - they construct tiny cleaned
matches/deliveries frames with hand-computed expected values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    MAX_RUN_RATE,
    WINDOW_BALLS,
    build_chase_features,
    latest_team_win_rates,
    overs_to_balls,
    rolling_team_win_rates,
)

BAT, BOWL = "Chasing XI", "Defending XI"


def make_match(match_id: int, date: str, season_year: int, winner: str, target_runs: int,
               target_overs: float = 20.0, result: str = "runs", team1: str = BOWL,
               team2: str = BAT) -> dict:
    """Build one cleaned matches.csv row (minimal columns used by features)."""
    return {
        "id": match_id, "date": date, "season_year": season_year, "venue": "Test Ground, Testville",
        "team1": team1, "team2": team2, "winner": winner, "result": result,
        "target_runs": float(target_runs), "target_overs": float(target_overs),
        "is_no_result": winner is None, "is_tie": False, "is_dls": False,
    }


def make_ball(match_id: int, inning: int, over: int, ball: int, runs: int,
              extras_type: str | None = None, wicket: str | None = None) -> dict:
    """Build one cleaned deliveries.csv row.

    ``wicket`` sets a genuine dismissal kind; ``extras_type`` 'wides' marks an
    illegal ball. Flags ``is_legal_ball`` / ``is_out`` are derived here exactly
    like src.data.clean_deliveries does.
    """
    illegal = extras_type in ("wides", "noballs")
    return {
        "match_id": match_id, "inning": inning, "over": over, "ball": ball,
        "batting_team": BAT, "bowling_team": BOWL,
        "total_runs": runs, "extras_type": extras_type,
        "is_wicket": int(wicket is not None or wicket == "retired hurt"),
        "dismissal_kind": wicket,
        "is_legal_ball": not illegal,
        "is_out": wicket is not None and wicket != "retired hurt",
    }


def make_matches(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def make_deliveries(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# overs_to_balls
# ---------------------------------------------------------------------------

def test_overs_to_balls_converts_cricket_notation() -> None:
    assert overs_to_balls(20.0) == 120
    assert overs_to_balls(9.2) == 56      # 9 overs + 2 balls, not 9.2 decimal
    assert overs_to_balls(0.1) == 1


def test_overs_to_balls_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        overs_to_balls(9.7)


# ---------------------------------------------------------------------------
# Rolling team win rates (no-future-leakage of team strength)
# ---------------------------------------------------------------------------

def test_rolling_win_rates_use_only_prior_matches() -> None:
    matches = make_matches([
        make_match(1, "2023-04-01", 2023, BAT, 100),
        make_match(2, "2023-04-02", 2023, BOWL, 100),
        make_match(3, "2023-04-03", 2023, BAT, 100),
    ])
    rates = rolling_team_win_rates(matches)
    bat = rates[rates["team"] == BAT].set_index("match_id")["prior_win_rate"]
    # match 1: first appearance -> NaN; match 2: 1 win from 1; match 3: 1 from 2
    # (BAT won match 3, but that win must NOT appear in match 3's own rate)
    assert np.isnan(bat.loc[1])
    assert bat.loc[2] == 1.0
    assert bat.loc[3] == 0.5


def test_latest_team_win_rates_takes_most_recent() -> None:
    matches = make_matches([
        make_match(1, "2023-04-01", 2023, BAT, 100),
        make_match(2, "2023-04-02", 2023, BAT, 100),
    ])
    latest = latest_team_win_rates(matches)
    assert latest[BAT] == 1.0  # two wins in a row


# ---------------------------------------------------------------------------
# build_chase_features on a tiny successful chase
# ---------------------------------------------------------------------------

def successful_chase() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Target 6 off 2 overs; chase wins with 1 run to spare in over 2."""
    matches = make_matches([make_match(1, "2023-04-01", 2023, BAT, 6, target_overs=2.0)])
    deliveries = make_deliveries([
        make_ball(1, 2, 0, 1, 1),
        make_ball(1, 2, 0, 2, 1, extras_type="wides"),   # wide: no legal ball
        make_ball(1, 2, 0, 3, 4),
        make_ball(1, 2, 1, 1, 1),                        # winning run
    ])
    return matches, deliveries


def test_basic_counts_and_rates() -> None:
    feats = build_chase_features(*successful_chase())
    assert len(feats) == 4

    first = feats.iloc[0]
    assert first["balls_bowled"] == 1 and first["runs_so_far"] == 1
    assert first["balls_left"] == 11 and first["runs_left"] == 5
    assert first["crr"] == pytest.approx(1 / (1 / 6))            # 6.0
    assert first["rrr"] == pytest.approx(5 / (11 / 6))           # 2.727...

    wide = feats.iloc[1]
    assert wide["balls_bowled"] == 1, "a wide must not advance the ball count"
    assert wide["runs_so_far"] == 2

    winning = feats.iloc[3]
    assert winning["runs_left"] == 0 and winning["balls_left"] == 9  # 12 - 3 legal balls
    assert winning["rrr"] == 0.0
    assert winning["chase_won"] == 1


def test_wickets_out_excludes_retired_hurt() -> None:
    matches = make_matches([make_match(1, "2023-04-01", 2023, BAT, 60, target_overs=20.0)])
    deliveries = make_deliveries([
        make_ball(1, 2, 0, 1, 0, wicket="caught"),
        make_ball(1, 2, 0, 2, 0, wicket="retired hurt"),
        make_ball(1, 2, 0, 3, 0),
    ])
    feats = build_chase_features(matches, deliveries)
    assert feats["wickets_out"].tolist() == [1, 1, 1], "retired hurt must not count as a wicket"
    assert feats["wickets_left"].tolist() == [9, 9, 9]


def test_last_window_sums() -> None:
    matches = make_matches([make_match(1, "2023-04-01", 2023, BAT, 60, target_overs=20.0)])
    balls = [make_ball(1, 2, 0, b, 1) for b in range(1, 7)]            # over 1: 6 runs
    balls += [make_ball(1, 2, 1, b, 2) for b in range(1, 7)]           # over 2: 12 runs
    balls += [make_ball(1, 2, 2, 1, 0, wicket="bowled")]               # over 3 ball 1
    feats = build_chase_features(matches, make_deliveries(balls))
    # window (30 balls) is longer than the innings so far -> full cumulative
    assert feats["last5_overs_runs"].tolist() == [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 18]
    assert feats["last5_overs_wickets"].tolist() == [0] * 12 + [1]


def test_rrr_clipped_at_physical_maximum_when_chase_fails() -> None:
    matches = make_matches([make_match(1, "2023-04-01", 2023, BOWL, 2, target_overs=1.0)])
    balls = [make_ball(1, 2, 0, b, 0) for b in range(1, 7)]            # all dots, 2 needed
    feats = build_chase_features(matches, make_deliveries(balls))
    last = feats.iloc[-1]
    assert last["balls_left"] == 0 and last["runs_left"] == 2
    assert last["rrr"] == MAX_RUN_RATE
    assert last["chase_won"] == 0


# ---------------------------------------------------------------------------
# Leakage checks: changing the future must not change the past
# ---------------------------------------------------------------------------

FEATURES_ONLY = FEATURE_COLUMNS  # all columns the model may see


def test_no_future_leakage_in_features() -> None:
    matches, deliveries = successful_chase()
    base = build_chase_features(matches, deliveries)

    # mutate the LAST ball: make it a six instead of a single
    deliveries_late = deliveries.copy()
    deliveries_late.loc[deliveries_late.index[-1], "total_runs"] = 6
    late = build_chase_features(matches, deliveries_late)

    earlier = [c for c in base.columns]
    pd.testing.assert_frame_equal(base.iloc[:3][earlier], late.iloc[:3][earlier])


def test_label_changes_without_feature_change() -> None:
    matches, deliveries = successful_chase()
    base = build_chase_features(matches, deliveries)

    matches_flipped = matches.copy()
    matches_flipped["winner"] = BOWL  # pretend the defence held
    flipped = build_chase_features(matches_flipped, deliveries)

    assert base["chase_won"].tolist() == [1, 1, 1, 1]
    assert flipped["chase_won"].tolist() == [0, 0, 0, 0]
    pd.testing.assert_frame_equal(base[FEATURES_ONLY], flipped[FEATURES_ONLY])


def test_ties_and_no_results_excluded() -> None:
    matches = make_matches([
        make_match(1, "2023-04-01", 2023, BAT, 6, target_overs=2.0),
        make_match(2, "2023-04-02", 2023, BAT, 6, target_overs=2.0, result="no result"),
        make_match(3, "2023-04-03", 2023, BAT, 6, target_overs=2.0, result="tie"),
    ])
    matches.loc[matches["id"] == 2, "is_no_result"] = True
    matches.loc[matches["id"] == 3, "is_tie"] = True
    balls = []
    for mid in (1, 2, 3):
        balls += [make_ball(mid, 2, 0, 1, 1)]
    feats = build_chase_features(matches, make_deliveries(balls))
    assert set(feats["match_id"]) == {1}


def test_window_constant_matches_feature_semantics() -> None:
    assert WINDOW_BALLS == 30  # "last 5 overs" = 30 legal balls
