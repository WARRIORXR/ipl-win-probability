"""Load, inspect and clean the raw IPL matches/deliveries data.

Data source
-----------
Kaggle IPL dataset ("matches" and "deliveries" files, 2008-2024 edition,
1,095 matches / 260,920 deliveries). Place the two CSVs in ``data/raw/``
(see README for download instructions). The files are gitignored.

This module defines every cleaning rule in explicit mapping dictionaries so
each decision is visible and auditable. Run ``python -m src.data clean`` to
write cleaned files to ``data/processed/`` or ``python -m src.data report``
to print a data-quality report.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

MATCHES_CSV = DATA_RAW / "matches.csv"
DELIVERIES_CSV = DATA_RAW / "deliveries.csv"

# ---------------------------------------------------------------------------
# Mapping dictionaries (all rules are explicit, no silent guessing)
# ---------------------------------------------------------------------------

#: Kaggle labels three seasons with span strings; map them to the year the
#: season was actually played (validated against match dates in clean_matches).
SEASON_ALIASES: dict[str, int] = {
    "2007/08": 2008,
    "2009": 2009,
    "2009/10": 2010,
    "2011": 2011,
    "2012": 2012,
    "2013": 2013,
    "2014": 2014,
    "2015": 2015,
    "2016": 2016,
    "2017": 2017,
    "2018": 2018,
    "2019": 2019,
    "2020/21": 2020,
    "2021": 2021,
    "2022": 2022,
    "2023": 2023,
    "2024": 2024,
}

#: Franchise renames and spelling variants -> one canonical name per franchise.
#: * Delhi Daredevils were renamed Delhi Capitals (2019).
#: * Kings XI Punjab were renamed Punjab Kings (2021).
#: * Royal Challengers Bangalore were renamed Royal Challengers Bengaluru (2024).
#: * "Rising Pune Supergiants" (2016) vs "Rising Pune Supergiant" (2017) are the
#:   same franchise; we keep the singular 2017 spelling.
#: * Deccan Chargers (2008-2012) became Sunrisers Hyderabad (2013-); they share
#:   the Hyderabad slot, so we merge them for team-strength features. This is a
#:   modelling choice, not a legal continuity claim.
#: * Gujarat Lions (2016-17) and Gujarat Titans (2022-) are DIFFERENT franchises
#:   and are deliberately kept separate, as are Pune Warriors and RPS.
TEAM_ALIASES: dict[str, str] = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
}

#: Venue name variants -> canonical "<Ground>, <City>" name. Covers missing city
#: suffixes, punctuation drift, and grounds that were officially renamed
#: (Feroz Shah Kotla -> Arun Jaitley Stadium 2019; Sardar Patel/Motera ->
#: Narendra Modi Stadium 2021; Subrata Roy Sahara -> MCA Stadium Pune).
VENUE_ALIASES: dict[str, str] = {
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Uppal",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Uppal",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium, Abu Dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium, Abu Dhabi",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium, Dubai",
    "Sharjah Cricket Stadium": "Sharjah Cricket Stadium, Sharjah",
    "Eden Gardens": "Eden Gardens, Kolkata",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy, Mumbai",
    "Brabourne Stadium": "Brabourne Stadium, Mumbai",
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Himachal Pradesh Cricket Association Stadium": "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Nehru Stadium": "Nehru Stadium, Kochi",
}

#: City spelling/usage fixes. The PCA stadium sits in Mohali (the dataset uses
#: "Chandigarh" for 61 of its 66 matches there), and Bengaluru is the official
#: spelling used elsewhere in the same column.
CITY_ALIASES: dict[str, str] = {
    "Bangalore": "Bengaluru",
    "Chandigarh": "Mohali",
}

#: Imputation for the 51 matches with a missing city (all 2021 UAE leg).
#: Keys are canonical (post-VENUE_ALIASES) venue names.
CITY_BY_VENUE: dict[str, str] = {
    "Dubai International Cricket Stadium, Dubai": "Dubai",
    "Sharjah Cricket Stadium, Sharjah": "Sharjah",
}

#: Extras that do NOT count as a legal ball. Byes/legbyes/penalty are scored
#: on legal deliveries, so they keep the ball alive.
NON_LEGAL_EXTRAS: frozenset[str] = frozenset({"wides", "noballs"})

#: Dismissals that do NOT count as a wicket: the batter may resume innings
#: later. The raw ``is_wicket`` flag wrongly counts these as outs.
NON_OUT_DISMISSALS: frozenset[str] = frozenset({"retired hurt"})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matches(path: Path | None = None) -> pd.DataFrame:
    """Read the raw matches CSV.

    Args:
        path: Override path; defaults to ``data/raw/matches.csv``.

    Returns:
        Raw matches dataframe (one row per match).
    """
    return pd.read_csv(path or MATCHES_CSV, low_memory=False)


def load_deliveries(path: Path | None = None) -> pd.DataFrame:
    """Read the raw ball-by-ball deliveries CSV.

    Args:
        path: Override path; defaults to ``data/raw/deliveries.csv``.

    Returns:
        Raw deliveries dataframe (one row per delivery, including super overs
        as ``inning`` >= 3).
    """
    return pd.read_csv(path or DELIVERIES_CSV, low_memory=False)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _map_series(series: pd.Series, mapping: dict[str, str], label: str) -> pd.Series:
    """Apply a rename mapping and log how many values changed."""
    mapped = series.map(lambda v: mapping.get(v, v))
    n_changed = int((mapped != series).sum())
    logger.info("%s: renamed %d values via %d mapping rules", label, n_changed, len(mapping))
    return mapped


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean the matches table without dropping any row.

    Adds ``season_year`` (int, validated against match dates), applies
    canonical team/venue/city names, imputes missing cities from the venue,
    and adds boolean flags ``is_no_result``, ``is_tie``, ``is_dls`` used by
    later phases to filter matches. Row count is preserved.

    Args:
        matches: Raw matches dataframe.

    Returns:
        Cleaned copy of the matches dataframe.
    """
    m = matches.copy()

    m["season_year"] = m["season"].map(SEASON_ALIASES)
    if m["season_year"].isna().any():
        unknown = sorted(m.loc[m["season_year"].isna(), "season"].unique())
        raise ValueError(f"Unmapped season labels: {unknown}")

    # The dataset labels some seasons with span strings; the year the season
    # was played must equal the match-date year (all IPL seasons here run
    # March-June, or Sept-Nov for 2020).
    date_year = pd.to_datetime(m["date"]).dt.year
    bad = m.loc[m["season_year"] != date_year, ["id", "season", "date"]]
    if not bad.empty:
        raise ValueError(f"season_year disagrees with match date for ids: {bad['id'].tolist()}")

    m["team1"] = _map_series(m["team1"], TEAM_ALIASES, "team1")
    m["team2"] = _map_series(m["team2"], TEAM_ALIASES, "team2")
    m["toss_winner"] = _map_series(m["toss_winner"], TEAM_ALIASES, "toss_winner")
    m["winner"] = _map_series(m["winner"], TEAM_ALIASES, "winner")
    m["venue"] = _map_series(m["venue"], VENUE_ALIASES, "venue")

    m["city"] = m["city"].fillna(m["venue"].map(CITY_BY_VENUE))
    n_imputed = int(m["city"].isna().sum())
    if n_imputed:
        raise ValueError(f"{n_imputed} cities still missing after venue-based imputation")
    m["city"] = _map_series(m["city"], CITY_ALIASES, "city")

    m["is_no_result"] = m["winner"].isna()
    m["is_tie"] = m["result"].eq("tie")
    m["is_dls"] = m["method"].eq("D/L")
    return m


def clean_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Clean the ball-by-ball table without dropping any row.

    Applies canonical team names and adds three flags used throughout the
    project: ``is_legal_ball`` (wides and no-balls do not advance the ball
    count), ``is_out`` (the raw ``is_wicket`` flag wrongly counts
    "retired hurt", which is not a dismissal) and ``is_super_over`` (raw
    ``inning`` >= 3 covers super overs, including the one double super over
    in the data, match 1216517).

    Args:
        deliveries: Raw deliveries dataframe.

    Returns:
        Cleaned copy of the deliveries dataframe.
    """
    d = deliveries.copy()

    d["batting_team"] = _map_series(d["batting_team"], TEAM_ALIASES, "batting_team")
    d["bowling_team"] = _map_series(d["bowling_team"], TEAM_ALIASES, "bowling_team")

    d["is_legal_ball"] = ~d["extras_type"].isin(NON_LEGAL_EXTRAS)
    d["is_out"] = d["is_wicket"].eq(1) & ~d["dismissal_kind"].isin(NON_OUT_DISMISSALS)
    d["is_super_over"] = d["inning"].ge(3)
    return d


# ---------------------------------------------------------------------------
# Reporting and pipeline entry point
# ---------------------------------------------------------------------------

def data_report() -> dict[str, object]:
    """Print a compact data-quality report of the raw files.

    Returns:
        Dict with row counts and the problem lists surfaced during Phase 0.
    """
    m = load_matches()
    d = load_deliveries()

    print("matches:", m.shape, "| deliveries:", d.shape)
    print("seasons:", sorted(m["season"].astype(str).unique()))
    print("raw team variants:", len(set(m["team1"]) | set(m["team2"])))
    print("raw venue variants:", m["venue"].nunique())
    print("missing city:", int(m["city"].isna().sum()),
          "| no-result matches:", int(m["winner"].isna().sum()),
          "| ties:", int(m["result"].eq("tie").sum()),
          "| D/L matches:", int(m["method"].notna().sum()))
    print("matches with reduced target_overs (rain/DLS):",
          int((m["target_overs"].notna() & m["target_overs"].ne(20)).sum()))
    print("deliveries with inning>=3 (super overs):", int(d["inning"].ge(3).sum()))
    print("'retired hurt' rows wrongly flagged is_wicket=1:",
          int((d["dismissal_kind"].eq("retired hurt") & d["is_wicket"].eq(1)).sum()))
    return {
        "matches_shape": m.shape,
        "deliveries_shape": d.shape,
        "n_team_variants": len(set(m["team1"]) | set(m["team2"])),
        "n_venue_variants": int(m["venue"].nunique()),
    }


def build_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline and write processed CSVs.

    Returns:
        Tuple of (cleaned matches, cleaned deliveries).
    """
    m = clean_matches(load_matches())
    d = clean_deliveries(load_deliveries())

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    m_path = DATA_PROCESSED / "matches_clean.csv"
    d_path = DATA_PROCESSED / "deliveries_clean.csv"
    m.to_csv(m_path, index=False)
    d.to_csv(d_path, index=False)

    print(f"wrote {m_path}  {m.shape}")
    print(f"wrote {d_path}  {d.shape}")
    print("canonical teams:", sorted(set(m['team1']) | set(m['team2'])))
    print("canonical venues:", m['venue'].nunique())
    print("canonical cities:", m['city'].nunique())
    return m, d


def main() -> None:
    """CLI entry point: ``python -m src.data report|clean``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="IPL data loading and cleaning")
    parser.add_argument("command", choices=["report", "clean"], help="report data quality or write cleaned CSVs")
    args = parser.parse_args()
    if args.command == "report":
        data_report()
    else:
        build_processed()


if __name__ == "__main__":
    main()
