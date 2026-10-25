"""Ball-by-ball feature engineering for 2nd-innings win probability.

Every feature uses only information available at the prediction point, which
we define as the state immediately AFTER the listed delivery has been bowled.
Nothing from later balls, the super over, or the match result (beyond the
label) enters the features.

Inputs are the CLEANED tables from :mod:`src.data` (canonical team names,
``is_legal_ball`` / ``is_out`` flags present).

Key conventions
---------------
* ``legal_balls_bowled`` counts only legal deliveries (wides and no-balls do
  not advance the count; byes/legbyes/penalty do).
* ``target`` is the actual target faced (``target_runs`` from matches.csv),
  which is DLS-adjusted for rain-affected chases.
* ``target_overs`` uses cricket "O.V" notation (9.2 = 9 overs + 2 balls);
  :func:`overs_to_balls` converts it to a delivery count.
* Run rates are clipped to at most 36.0 runs per over (6 per ball), the
  physical maximum, to keep extreme tail values from distorting scaling.
* Ties (super-over deciders) and no-result matches are EXCLUDED from the
  modelling dataset so the label means "the chase was completed successfully
  in regulation play".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"

#: "Last 5 overs" momentum window, in legal deliveries.
WINDOW_BALLS = 30
#: Physical maximum runs per over; used to clip run rates.
MAX_RUN_RATE = 36.0
#: Wickets available to the chasing side.
TOTAL_WICKETS = 10

NUMERIC_FEATURES: list[str] = [
    "target",
    "runs_so_far",
    "runs_left",
    "balls_left",
    "balls_bowled",
    "wickets_out",
    "wickets_left",
    "crr",
    "rrr",
    "last5_overs_runs",
    "last5_overs_wickets",
    "rolling_win_rate_bat",
    "rolling_win_rate_bowl",
]
CATEGORICAL_FEATURES: list[str] = ["batting_team", "bowling_team", "venue"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
META_COLUMNS: list[str] = [
    "match_id",
    "season_year",
    "over",
    "ball",
    "target_overs",
    "date",
]
LABEL = "chase_won"


def overs_to_balls(overs: float) -> int:
    """Convert cricket "O.V" over notation to a number of deliveries.

    Args:
        overs: Overs in ``O.V`` notation, e.g. ``9.2`` means 9 overs and
            2 balls (56 deliveries), NOT 9.2 decimal overs.

    Returns:
        Number of deliveries.

    Raises:
        ValueError: If the ball fraction is outside 0-5, which would mean the
            value is not valid ``O.V`` notation.
    """
    whole = int(overs)
    balls = round((overs - whole) * 10)
    if not 0 <= balls <= 5:
        raise ValueError(f"{overs!r} is not valid O.V over notation")
    return whole * 6 + balls


def rolling_team_win_rates(matches: pd.DataFrame) -> pd.DataFrame:
    """Win rate of each team over its PRIOR matches only.

    For every match and every participating team, computes the expanding mean
    of the win indicator over strictly earlier matches (lagged by one), so the
    current match can never leak into its own feature. Ties decided by super
    over count as wins for the recorded winner; no-result matches are excluded.

    Args:
        matches: Cleaned matches table.

    Returns:
        Long dataframe with columns ``match_id``, ``team``, ``prior_win_rate``
        (NaN for a team's first match -> caller fills 0.5).
    """
    m = matches.loc[~matches["is_no_result"]].sort_values(["date", "id"], kind="stable")
    sides = []
    for side in ("team1", "team2"):
        sides.append(
            pd.DataFrame(
                {
                    "match_id": m["id"].to_numpy(),
                    "team": m[side].to_numpy(),
                    "won": (m["winner"] == m[side]).astype(float).to_numpy(),
                }
            )
        )
    long = pd.concat(sides, ignore_index=True).sort_values(["match_id", "team"], kind="stable")
    long["prior_win_rate"] = (
        long.groupby("team", sort=False)["won"].transform(lambda s: s.shift(1).expanding().mean())
    )
    return long[["match_id", "team", "prior_win_rate"]]


def latest_team_win_rates(matches: pd.DataFrame) -> dict[str, float]:
    """Most recent prior-match win rate per team (for the deployed app).

    Args:
        matches: Cleaned matches table.

    Returns:
        Mapping of canonical team name to its latest rolling win rate
        (teams never seen get 0.5 by convention at call sites).
    """
    long = rolling_team_win_rates(matches)
    long = long.dropna()
    latest = long.groupby("team")["prior_win_rate"].last()
    return {team: float(rate) for team, rate in latest.items()}


def _add_window_features(g: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative and last-30-legal-balls features to one match's chase.

    Args:
        g: Rows of one match's 2nd innings in chronological order.

    Returns:
        Copy of ``g`` with ``legal_balls_bowled``, ``runs_so_far``,
        ``wickets_out``, ``last5_overs_runs`` and ``last5_overs_wickets``.
    """
    g = g.sort_values(["over", "ball"], kind="stable")
    legal_balls = g["is_legal_ball"].cumsum().astype(int)
    runs_cum = g["total_runs"].cumsum()
    wkt_cum = g["is_out"].cumsum()

    # State after each legal ball number: the last row carrying that number
    # also carries any wide/no-ball rows attached to it (they share the count).
    state = (
        pd.DataFrame(
            {
                "lb": legal_balls.to_numpy(),
                "runs": runs_cum.to_numpy(),
                "wkts": wkt_cum.to_numpy(),
            }
        )
        .groupby("lb")
        .last()
    )
    start = (legal_balls - WINDOW_BALLS).clip(lower=0).to_numpy()
    prior_runs = state["runs"].reindex(start).fillna(0.0).to_numpy()
    prior_wkts = state["wkts"].reindex(start).fillna(0.0).to_numpy()

    out = g.copy()
    out["legal_balls_bowled"] = legal_balls.to_numpy()
    out["runs_so_far"] = runs_cum.to_numpy()
    out["wickets_out"] = wkt_cum.to_numpy()
    out["last5_overs_runs"] = runs_cum.to_numpy() - prior_runs
    out["last5_overs_wickets"] = wkt_cum.to_numpy() - prior_wkts
    return out


def build_chase_features(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    """Build the 2nd-innings ball-by-ball modelling dataset.

    Excludes no-result and tie (super-over) matches, keeps every ball of every
    remaining regulation 2nd innings, and computes all features from the state
    at that ball (see module docstring for conventions).

    Args:
        matches: Cleaned matches table from :func:`src.data.clean_matches`.
        deliveries: Cleaned deliveries table from :func:`src.data.clean_deliveries`.

    Returns:
        Feature dataframe with meta columns, features and the ``chase_won`` label.
    """
    m = matches.loc[~matches["is_no_result"] & ~matches["is_tie"]].copy()
    d = deliveries.loc[deliveries["inning"] == 2].copy()

    m_cols = m[["id", "date", "season_year", "venue", "winner", "target_runs", "target_overs"]]
    d = d.merge(
        m_cols,
        left_on="match_id",
        right_on="id",
        how="inner",
        validate="many_to_one",
    )
    if d[["target_runs", "target_overs"]].isna().any().any():
        raise ValueError("chase rows with missing target after filtering - check the cleaning step")

    d = pd.concat(
        [_add_window_features(g) for _, g in d.groupby("match_id", sort=True)],
        ignore_index=False,
    )

    balls_total = d["target_overs"].map(overs_to_balls)
    d["balls_bowled"] = d["legal_balls_bowled"].astype(int)
    d["balls_left"] = (balls_total - d["balls_bowled"]).clip(lower=0)
    d["runs_left"] = (d["target_runs"] - d["runs_so_far"]).clip(lower=0)
    d["wickets_left"] = (TOTAL_WICKETS - d["wickets_out"]).clip(lower=0)

    balls_faced = d["balls_bowled"].astype(float)
    d["crr"] = np.where(
        balls_faced > 0, (d["runs_so_far"] / (balls_faced / 6.0)).clip(upper=MAX_RUN_RATE), 0.0
    )
    rrr_raw = np.where(
        d["balls_left"] > 0,
        d["runs_left"] / (d["balls_left"] / 6.0),
        np.where(d["runs_left"] > 0, MAX_RUN_RATE, 0.0),
    )
    d["rrr"] = np.clip(rrr_raw, 0.0, MAX_RUN_RATE)
    d["target"] = d["target_runs"].astype(int)

    rates = rolling_team_win_rates(matches)
    d = d.merge(
        rates.rename(columns={"team": "batting_team", "prior_win_rate": "rolling_win_rate_bat"}),
        on=["match_id", "batting_team"],
        how="left",
        validate="many_to_one",
    ).merge(
        rates.rename(columns={"team": "bowling_team", "prior_win_rate": "rolling_win_rate_bowl"}),
        on=["match_id", "bowling_team"],
        how="left",
        validate="many_to_one",
    )
    d["rolling_win_rate_bat"] = d["rolling_win_rate_bat"].fillna(0.5)
    d["rolling_win_rate_bowl"] = d["rolling_win_rate_bowl"].fillna(0.5)

    d[LABEL] = (d["batting_team"] == d["winner"]).astype(int)

    keep = META_COLUMNS + FEATURE_COLUMNS + [LABEL]
    return d[keep].reset_index(drop=True)


def main() -> None:
    """CLI entry point: ``python -m src.features`` writes the feature CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from src.data import clean_deliveries, clean_matches, load_deliveries, load_matches

    features = build_chase_features(clean_matches(load_matches()), clean_deliveries(load_deliveries()))
    out = DATA_PROCESSED / "chase_features.csv"
    features.to_csv(out, index=False)
    print(f"wrote {out}  {features.shape}")
    print("matches covered:", features["match_id"].nunique())
    print("seasons:", features["season_year"].min(), "-", features["season_year"].max())
    print("chase win rate:", round(features[LABEL].mean(), 4))
    print("label counts:", features[LABEL].value_counts().to_dict())


if __name__ == "__main__":
    main()
