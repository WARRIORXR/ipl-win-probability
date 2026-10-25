"""Tests for the season-based split in src/train.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.train import TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS, season_split


def _features(match_id: int, season: int) -> dict:
    """One minimal feature row."""
    return {"match_id": match_id, "season_year": season, "chase_won": 1}


def test_split_covers_all_seasons_without_overlap() -> None:
    features = pd.DataFrame(
        [_features(match_id=season * 10 + i, season=season)
         for season in range(2008, 2025) for i in range(2)]
    )
    parts = season_split(features)
    assert set(parts) == {"train", "validation", "test"}
    assert parts["train"]["season_year"].isin(TRAIN_SEASONS).all()
    assert parts["validation"]["season_year"].isin(VAL_SEASONS).all()
    assert parts["test"]["season_year"].isin(TEST_SEASONS).all()
    all_ids = pd.concat([p["match_id"] for p in parts.values()])
    assert all_ids.is_unique, "no match may appear in two splits"
    assert len(all_ids) == len(features)


def test_split_rejects_match_spanning_seasons() -> None:
    """A match id seen in two season buckets must raise."""
    features = pd.DataFrame([_features(1, 2022), _features(1, 2023)])
    with pytest.raises(ValueError, match="more than one split"):
        season_split(features)


def test_split_boundaries_are_the_documented_ones() -> None:
    assert TRAIN_SEASONS == list(range(2008, 2022))
    assert VAL_SEASONS == [2022]
    assert TEST_SEASONS == [2023, 2024]
