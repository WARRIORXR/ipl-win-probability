"""IPL Win Probability - Streamlit app.

Estimates the batting team's win probability at any point of a 2nd-innings
chase, with a match-replay view over real 2023-24 chases.

Run from the repository root:  streamlit run app/app.py

This app is for learning and portfolio demonstration only - it is NOT for
betting or gambling. Predictions are statistical estimates from historical
data, not advice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.predict import predict_situation  # noqa: E402

st.set_page_config(page_title="IPL Win Probability", page_icon="🏏", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model() -> dict:
    """Load model artifacts once per app session (Streamlit-cached)."""
    from src.predict import load_model

    return load_model()


@st.cache_data(show_spinner="Loading match data...")
def get_replay_data() -> pd.DataFrame:
    """Load the committed per-ball dataset of 2023-24 chases."""
    return pd.read_csv(ROOT / "app" / "assets" / "replay_chases.csv")


@st.cache_data(show_spinner=False)
def get_importance() -> dict[str, float]:
    """Top feature importances from the training metadata."""
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return dict(list(meta["feature_importance"].items())[:6])


ART = get_model()
#: Only franchises active in the test seasons are offered: the linear model's
#: team effects are one-hot coefficients learned on 2008-21, and pairing an
#: active team with a defunct one (e.g. CSK vs Gujarat Lions, which never
#: happened - CSK was suspended 2016-17) extrapolates nonsense.
REPLAY = get_replay_data()
ACTIVE_TEAMS: list[str] = sorted(
    set(REPLAY["batting_team"]) | set(REPLAY["bowling_team"])
)
VENUES: list[str] = ART["venues"]

st.title("🏏 IPL Win Probability")
st.caption(
    "Win probability for the batting team at any point of a 20-over chase. "
    "Logistic regression trained on ball-by-ball IPL data: seasons 2008-2021 "
    "(train), 2022 (validation), 2023-24 (test)."
)
st.warning(
    "**Disclaimer:** for learning and portfolio demonstration only. "
    "**Not for betting or gambling** - predictions are statistical estimates "
    "from historical data, not advice.",
    icon="⚠️",
)

live_tab, replay_tab = st.tabs(["Live situation", "Match replay"])

# ---------------------------------------------------------------------------
# Live situation
# ---------------------------------------------------------------------------

with live_tab:
    col_team1, col_team2 = st.columns(2)
    batting = col_team1.selectbox(
        "Batting team (chasing)", ACTIVE_TEAMS,
        index=ACTIVE_TEAMS.index("Chennai Super Kings") if "Chennai Super Kings" in ACTIVE_TEAMS else 0,
        key="batting",
    )
    bowling_options = [t for t in ACTIVE_TEAMS if t != batting]
    bowling = col_team2.selectbox(
        "Bowling team (defending)", bowling_options,
        index=bowling_options.index("Mumbai Indians") if "Mumbai Indians" in bowling_options else 0,
        key="bowling",
    )
    venue = st.selectbox("Venue", VENUES, key="venue")
    st.caption("Only franchises active in 2023-24 are offered - the model's team "
               "effects must not be extrapolated to defunct pairings that never occurred.")

    col_t, col_s, col_w = st.columns(3)
    target = col_t.number_input("Target", min_value=1, max_value=300, value=180, key="target")
    score = col_s.number_input("Current score", min_value=0, max_value=400, value=90, key="score")
    wickets = col_w.number_input("Wickets lost", min_value=0, max_value=10, value=3, key="wickets")

    col_o, col_b = st.columns(2)
    overs = col_o.number_input("Overs completed", min_value=0, max_value=19, value=10, key="overs")
    balls = col_b.number_input("Balls into current over", min_value=0, max_value=5, value=3, key="balls")

    errors: list[str] = []
    if batting == bowling:
        errors.append("The batting and bowling teams must be different.")
    if score >= target:
        errors.append(
            f"Score {score:.0f} already reaches the target of {target:.0f} - "
            "the chase would be over. Enter a score below the target."
        )
    if overs * 6 + balls > 120:
        errors.append("A 20-over chase has at most 120 legal balls (20 overs).")
    if errors:
        for message in errors:
            st.error(message)
        st.stop()

    result = predict_situation(
        batting_team=batting,
        bowling_team=bowling,
        venue=venue,
        target=int(target),
        score=int(score),
        overs_completed=int(overs),
        balls_completed=int(balls),
        wickets_lost=int(wickets),
    )
    p = result["p_bat"]

    st.subheader(f"{batting} win probability: {p * 100:.1f}%")
    st.progress(
        p,
        text=f"{batting} {p * 100:.1f}%  |  {bowling} {(1 - p) * 100:.1f}%",
    )

    m_crr, m_rrr, m_runs, m_balls = st.columns(4)
    m_crr.metric("Current run rate", f"{result['crr']:.2f}")
    m_rrr.metric("Required run rate", f"{result['rrr']:.2f}")
    m_runs.metric("Runs left", f"{result['runs_left']:.0f}")
    m_balls.metric("Balls left", f"{result['balls_left']:.0f}")

    with st.expander("What drives this prediction"):
        importance = get_importance()
        st.markdown(
            "The model weighs, in order: "
            + ", ".join(f"**{k}** ({v:.0%})" for k, v in importance.items())
            + ". For this situation the key numbers are a required rate of "
            f"**{result['rrr']:.2f}** against a current rate of **{result['crr']:.2f}**, "
            f"with **{10 - int(wickets)} wickets in hand** and "
            f"**{result['balls_left']:.0f} balls left**."
        )
        rows = []
        score_up = min(int(score) + 6, int(target) - 1)
        if result["runs_left"] > 0:
            alt = predict_situation(batting, bowling, venue, int(target), score_up,
                                    int(overs), int(balls), int(wickets))
            rows.append(("+6 runs (a big over)", alt["p_bat"]))
        if wickets < 10:
            alt = predict_situation(batting, bowling, venue, int(target), int(score),
                                    int(overs), int(balls), int(wickets) + 1)
            rows.append(("one more wicket falls", alt["p_bat"]))
        balls_bowled = int(overs) * 6 + int(balls)
        if result["balls_left"] >= 6:
            new_bb = balls_bowled + 6
            alt = predict_situation(batting, bowling, venue, int(target), int(score),
                                    new_bb // 6, new_bb % 6, int(wickets))
            rows.append(("6 more dot balls", alt["p_bat"]))
        if rows:
            what_if = pd.DataFrame(
                [{"scenario": name, "win probability": f"{prob * 100:.1f}%"} for name, prob in rows]
            )
            st.table(what_if.set_index("scenario"))
        st.caption("Scenario probabilities move the situation and re-predict; "
                   "they assume the same venue and teams.")

# ---------------------------------------------------------------------------
# Match replay
# ---------------------------------------------------------------------------

with replay_tab:
    st.markdown("Win probability through real chases from the test seasons (2023-24).")
    meta = REPLAY.drop_duplicates("match_id").set_index("match_id")

    def match_label(match_id: int) -> str:
        row = meta.loc[match_id]
        return (f"{int(row['season_year'])}: {row['batting_team']} chasing "
                f"{int(row['target'])} vs {row['bowling_team']}")

    match_ids = sorted(meta.index, key=match_label)
    choice = st.selectbox("Pick a match", match_ids, format_func=match_label, key="replay_match")
    demo = REPLAY[REPLAY["match_id"] == choice].sort_values(["over", "ball"])
    probs = ART["pipeline"].predict_proba(demo[ART["feature_columns"]])[:, 1]
    wicket_balls = demo.loc[demo["wickets_out"].diff() > 0, "balls_bowled"].tolist()

    fig = ev.plot_win_curve(
        demo["balls_bowled"].tolist(),
        probs.tolist(),
        wicket_balls,
        title=match_label(choice),
    )
    st.pyplot(fig)

    outcome = "chase won" if demo["chase_won"].iloc[-1] == 1 else "chase lost"
    st.caption(f"Final outcome: **{outcome}**. Red crosses mark wickets; the "
               "dashed line is 50%. Each point uses only information available at that ball.")

st.divider()
st.caption(
    "Model: logistic regression (season split 2008-21 / 2022 / 2023-24) · "
    "Data: Kaggle IPL matches & deliveries · Built for learning and portfolio use - "
    "**not for betting or gambling**."
)
