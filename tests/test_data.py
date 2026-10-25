"""Tests for src/data.py cleaning rules.

These tests read the real raw CSVs and are skipped when the data files are
absent (e.g. a fresh clone before downloading the data).
"""

from pathlib import Path

import pandas as pd
import pytest

from src import data as data_mod

pytestmark = pytest.mark.skipif(
    not (data_mod.MATCHES_CSV.exists() and data_mod.DELIVERIES_CSV.exists()),
    reason="raw data files not present in data/raw/",
)


@pytest.fixture(scope="module")
def matches_clean() -> pd.DataFrame:
    """Cleaned matches table for the module."""
    return data_mod.clean_matches(data_mod.load_matches())


@pytest.fixture(scope="module")
def deliveries_clean() -> pd.DataFrame:
    """Cleaned deliveries table for the module."""
    return data_mod.clean_deliveries(data_mod.load_deliveries())


def test_no_rows_dropped_by_cleaning(matches_clean: pd.DataFrame, deliveries_clean: pd.DataFrame) -> None:
    """Cleaning renames and flags; it must never drop rows."""
    assert len(matches_clean) == 1095
    assert len(deliveries_clean) == 260920


def test_teams_map_to_canonical_names(matches_clean: pd.DataFrame, deliveries_clean: pd.DataFrame) -> None:
    """After cleaning, only canonical franchise names remain."""
    expected = {
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Lions", "Gujarat Titans",
        "Kochi Tuskers Kerala", "Kolkata Knight Riders", "Lucknow Super Giants",
        "Mumbai Indians", "Punjab Kings", "Rajasthan Royals", "Rising Pune Supergiant",
        "Pune Warriors", "Royal Challengers Bengaluru", "Sunrisers Hyderabad",
    }
    match_teams = set(matches_clean["team1"]) | set(matches_clean["team2"]) | set(matches_clean["winner"].dropna())
    ball_teams = set(deliveries_clean["batting_team"]) | set(deliveries_clean["bowling_team"])
    assert match_teams == expected, match_teams - expected
    assert ball_teams == expected, ball_teams - expected


def test_venue_variants_are_collapsed(matches_clean: pd.DataFrame) -> None:
    """58 raw venue spellings must collapse to a smaller canonical set."""
    n_canonical = matches_clean["venue"].nunique()
    assert n_canonical < 40, n_canonical


def test_no_missing_cities_and_known_renames_fixed(matches_clean: pd.DataFrame) -> None:
    """Cities are fully populated and the two known spellings are merged."""
    assert matches_clean["city"].isna().sum() == 0
    cities = set(matches_clean["city"])
    assert "Bangalore" not in cities and "Bengaluru" in cities
    assert "Chandigarh" not in cities


def test_season_year_matches_match_date(matches_clean: pd.DataFrame) -> None:
    """Span-labelled seasons (2007/08, 2009/10, 2020/21) map to the played year."""
    date_year = pd.to_datetime(matches_clean["date"]).dt.year
    assert (matches_clean["season_year"] == date_year).all()


def test_retired_hurt_not_counted_as_wicket(deliveries_clean: pd.DataFrame) -> None:
    """is_out must exclude retired hurt, which the raw is_wicket flag counts."""
    retired = deliveries_clean["dismissal_kind"].eq("retired hurt")
    assert (deliveries_clean.loc[retired, "is_out"] == 0).all()
    # every raw is_wicket=1 row that is a genuine dismissal stays an out
    genuine = deliveries_clean["is_wicket"].eq(1) & ~retired
    assert (deliveries_clean.loc[genuine, "is_out"] == 1).all()


def test_wides_and_noballs_not_legal_balls(deliveries_clean: pd.DataFrame) -> None:
    """Only wides/no-balls are illegal; byes and leg-byes keep the ball alive."""
    illegal = deliveries_clean[~deliveries_clean["is_legal_ball"]]
    assert set(illegal["extras_type"].dropna().unique()) == {"wides", "noballs"}
    byes = deliveries_clean["extras_type"].isin(["byes", "legbyes", "penalty"])
    assert (deliveries_clean.loc[byes, "is_legal_ball"]).all()
