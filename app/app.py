"""IPL Win Probability - Streamlit app.

Estimates the batting team's win probability at any point of a 2nd-innings
chase, with a match-replay view over real 2023-24 chases.

Run from the repository root:  streamlit run app/app.py

This app is for learning and portfolio demonstration only - it is NOT for
betting or gambling. Predictions are statistical estimates from historical
data, not advice.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set by src.evaluate)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.predict import predict_situation  # noqa: E402

st.set_page_config(page_title="IPL Win Probability", page_icon="🏏", layout="centered")

#: Approximate franchise brand colours (canonical team name -> hex).
TEAM_COLORS: dict[str, str] = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#DA1818",
    "Kolkata Knight Riders": "#5C2D91",
    "Sunrisers Hyderabad": "#F26522",
    "Rajasthan Royals": "#EA1A85",
    "Delhi Capitals": "#17479E",
    "Punjab Kings": "#D71920",
    "Gujarat Titans": "#1B2C5C",
    "Lucknow Super Giants": "#0072BC",
}
TEAM_SHORT: dict[str, str] = {
    "Chennai Super Kings": "CSK",
    "Mumbai Indians": "MI",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Rajasthan Royals": "RR",
    "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
}
DEFAULT_COLOR = "#1F77B4"

#: One-click example situations (label -> widget values).
PRESETS: dict[str, dict[str, int]] = {
    "😌 Steady chase": {"target": 180, "score": 95, "wickets": 2, "overs": 12, "balls": 0},
    "😰 Wickets clattering": {"target": 180, "score": 78, "wickets": 5, "overs": 12, "balls": 0},
    "🔥 Final-over thriller": {"target": 190, "score": 176, "wickets": 5, "overs": 19, "balls": 3},
}

CUSTOM_CSS = """
<style>
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 12px 16px 10px 16px;
    }
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    [data-testid="stMetricLabel"] p {font-size: 0.86rem !important;}
    [data-testid="stTab"] button {font-size: 1.02rem; padding: 10px 4px;}
    .winbar {display: flex; width: 100%; height: 46px; border-radius: 11px;
             overflow: hidden; font-weight: 700; font-size: 0.92rem;
             box-shadow: 0 1px 6px rgba(0,0,0,0.35);}
    .winbar .seg {display: flex; align-items: center; justify-content: center;
                  color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.55);
                  white-space: nowrap; overflow: hidden;}
    .scoreboard {display: flex; gap: 10px; flex-wrap: wrap; margin: 2px 0 10px 0;}
    .chip {background: rgba(255,255,255,0.045); border: 1px solid rgba(255,255,255,0.09);
           border-radius: 10px; padding: 8px 16px; display: flex; flex-direction: column;
           align-items: center; min-width: 88px;}
    .chip .lbl {font-size: 0.66rem; letter-spacing: 0.09em; color: #9AA0A6;}
    .chip .val {font-size: 1.3rem; font-weight: 700; line-height: 1.25;}
    .verdict {border-left: 4px solid #888; padding: 8px 14px; margin: 8px 0 4px 0;
              background: rgba(255,255,255,0.04); border-radius: 6px;
              font-size: 0.95rem;}
    div[data-testid="stExpander"] {border-radius: 12px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def team_color(team: str) -> str:
    """Franchise colour with a neutral fallback for unknown names."""
    return TEAM_COLORS.get(team, DEFAULT_COLOR)


def team_short(team: str) -> str:
    """Short franchise code for tight UI labels."""
    return TEAM_SHORT.get(team, team[:3].upper())


def draw_gauge(p: float, color: str):
    """Semicircular win-probability gauge in the batting team's colour.

    Args:
        p: Win probability of the batting team (0-1).
        color: Arc colour.

    Returns:
        The Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(3.1, 2.0), dpi=150)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.26, facecolor="#2A2F3A"))
    ax.add_patch(Wedge((0, 0), 1.0, 180 * (1 - p), 180, width=0.26, facecolor=color))
    ax.text(0, 0.10, f"{p * 100:.1f}%", ha="center", va="center",
            fontsize=21, fontweight="bold", color="#FAFAFA")
    ax.text(0, -0.16, "win chance", ha="center", va="center",
            fontsize=8.5, color="#9AA0A6")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.30, 1.12)
    ax.axis("off")
    fig.tight_layout()
    return fig


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
def get_meta() -> dict:
    """Training metadata (metrics, split, importance) for the model card."""
    return json.loads((ROOT / "models" / "metadata.json").read_text())


@st.cache_data(show_spinner=False)
def get_importance() -> dict[str, float]:
    """Top feature importances from the training metadata."""
    return dict(list(get_meta()["feature_importance"].items())[:6])


@st.cache_data(show_spinner=False)
def get_venue_stats() -> pd.DataFrame:
    """Historical chase success rate per venue (2008-24), from the features data."""
    path = ROOT / "app" / "assets" / "venue_chase_stats.csv"
    return pd.read_csv(path).set_index("venue")


def chase_outlook(result: dict, target: int, score: int, overs: int, balls: int,
                  wickets: int, batting: str, bowling: str, venue: str) -> tuple[pd.DataFrame, int]:
    """Project P(win) over the coming overs under three scoring-rate scenarios.

    Scenarios: keep the current run rate, score exactly at the required rate
    (which completes the chase by definition), or fall 2 runs per over behind
    the requirement. Wickets are held constant - stated in the caption. All
    scenario rows are predicted in ONE vectorized pipeline call (profiled:
    33 single-row calls ~110 ms -> one batch ~5 ms).

    Args:
        result: Output of :func:`src.predict.predict_situation` for the now-state.
        target / score / overs / balls / wickets: Current situation inputs.
        batting / bowling / venue: Situation labels.

    Returns:
        (long dataframe with columns scenario / balls_from_now / p, steps projected).
    """
    balls_left = result["balls_left"]
    steps = min(10, max(1, -(-balls_left // 6)))  # ceil division, at least one step
    scenarios = [
        ("keep current rate", result["crr"]),
        ("score at required rate", result["rrr"]),
        ("2 rpo slower than needed", max(result["rrr"] - 2.0, 0.0)),
    ]
    strength = ART["team_strength"]
    rows: list[dict] = []
    meta_rows: list[tuple[str, int, int | None]] = []  # scenario, balls_from_now, runs_added
    for label, rate in scenarios:
        for k in range(steps + 1):
            balls_used = min(6 * k, balls_left)
            runs_added = min(round(rate * balls_used / 6.0), result["runs_left"])
            meta_rows.append((label, balls_used, None if result["runs_left"] - runs_added <= 0 else runs_added))
            rows.append({
                "target": int(target),
                "runs_so_far": min(score + runs_added, target - 1),
                "runs_left": max(result["runs_left"] - runs_added, 0),
                "balls_left": max(balls_left - balls_used, 0),
                "balls_bowled": overs * 6 + balls + balls_used,
                "wickets_out": int(wickets),
                "wickets_left": 10 - int(wickets),
                "crr": min((score + runs_added) / max((overs * 6 + balls + balls_used) / 6.0, 1e-9), 36.0)
                       if overs * 6 + balls + balls_used > 0 else 0.0,
                "rrr": (min((result["runs_left"] - runs_added) / max((balls_left - balls_used) / 6.0, 1e-9), 36.0)
                        if balls_left - balls_used > 0 else (36.0 if result["runs_left"] - runs_added > 0 else 0.0)),
                "rolling_win_rate_bat": strength.get(batting, 0.5),
                "rolling_win_rate_bowl": strength.get(bowling, 0.5),
                "batting_team": batting,
                "bowling_team": bowling,
                "venue": venue,
            })
    X = pd.DataFrame(rows)[ART["feature_columns"]]
    probs = ART["pipeline"].predict_proba(X)[:, 1]
    calibrator = ART["calibrator"]
    if calibrator is not None:  # deployed artifact currently ships uncalibrated
        if ART["calibrator_name"] == "isotonic":
            probs = calibrator.predict(probs)
        else:
            probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    out = [
        {"scenario": label, "balls from now": balls_used,
         "p": 1.0 if runs_added is None else float(prob)}
        for (label, balls_used, runs_added), prob in zip(meta_rows, probs)
    ]
    return pd.DataFrame(out), steps


@st.cache_data(show_spinner=False)
def replay_curve_png(match_id: int) -> bytes:
    """Render one match's replay curve once and cache the PNG bytes.

    The full render (batch prediction + matplotlib figure) is ~45 ms; caching
    by match id makes repeat selections and tab switches free (profiled in
    scripts/profile_app.py).

    Args:
        match_id: Match id from the committed replay dataset.

    Returns:
        PNG bytes of the win-probability curve.
    """
    meta_row = REPLAY.drop_duplicates("match_id").set_index("match_id").loc[match_id]
    demo = REPLAY[REPLAY["match_id"] == match_id].sort_values(["over", "ball"])
    probs = ART["pipeline"].predict_proba(demo[ART["feature_columns"]])[:, 1]
    wicket_balls = demo.loc[demo["wickets_out"].diff() > 0, "balls_bowled"].tolist()
    title = (f"{int(meta_row['season_year'])}: {meta_row['batting_team']} chasing "
             f"{int(meta_row['target'])} vs {meta_row['bowling_team']} · "
             f"{meta_row['venue'].split(', ')[-1]}")
    fig = ev.plot_win_curve(
        demo["balls_bowled"].tolist(), probs.tolist(), wicket_balls,
        color=team_color(meta_row["batting_team"]), title=title,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


ART = get_model()
#: Only franchises active in the test seasons are offered: the linear model's
#: team effects are one-hot coefficients learned on 2008-21, and pairing an
#: active team with a defunct one (e.g. CSK vs Gujarat Lions, which never
#: happened - CSK was suspended 2016-17) extrapolates nonsense.
REPLAY = get_replay_data()
ACTIVE_TEAMS: list[str] = sorted(set(REPLAY["batting_team"]) | set(REPLAY["bowling_team"]))
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

live_tab, replay_tab, about_tab = st.tabs(["⚡ Live situation", "🎬 Match replay", "📊 Model card"])

# ---------------------------------------------------------------------------
# Live situation
# ---------------------------------------------------------------------------

with live_tab:
    with st.container(border=True):
        st.markdown("##### 🏟️ Match situation")
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
        col_team1.caption(f"Recent form: {ART['team_strength'].get(batting, 0.5):.0%} wins "
                          "(rolling, through 2024)")
        col_team2.caption(f"Recent form: {ART['team_strength'].get(bowling, 0.5):.0%} wins "
                          "(rolling, through 2024)")
        venue = st.selectbox("Venue", VENUES, key="venue",
                             help="Where the match is played - the model's heaviest feature.")
        venue_stats = get_venue_stats()
        if venue in venue_stats.index:
            vrow = venue_stats.loc[venue]
            st.caption(f"🏟️ Chasing teams win {vrow['chase_win_rate']:.0%} of matches at this venue "
                       f"({int(vrow['matches'])} matches, 2008-24).")
        st.caption("Only franchises active in 2023-24 are offered - the model's team "
                   "effects must not be extrapolated to defunct pairings that never occurred.")

        col_t, col_s, col_w = st.columns(3)
        target = col_t.number_input("Target", min_value=1, max_value=300, value=180, key="target",
                                    help="Total the batting team must pass to win.")
        score = col_s.number_input("Current score", min_value=0, max_value=400, value=90, key="score",
                                   help="Runs scored so far in the chase.")
        wickets = col_w.number_input("Wickets lost", min_value=0, max_value=10, value=3, key="wickets",
                                     help="Wickets down (10 all out).")
        col_o, col_b = st.columns(2)
        overs = col_o.number_input("Overs completed", min_value=0, max_value=19, value=10, key="overs",
                                   help="Full overs bowled in the chase so far.")
        balls = col_b.number_input("Balls into current over", min_value=0, max_value=5, value=3, key="balls",
                                   help="Legal balls bowled in the current over (0-5).")

        st.caption("⚡ Quick presets - one click fills the whole situation:")
        preset_row = st.columns(len(PRESETS))

        def apply_preset(values: dict[str, int]) -> None:
            """Button callback: runs before the rerun so widgets pick up values."""
            for widget_key, value in values.items():
                st.session_state[widget_key] = value

        for col, (label, values) in zip(preset_row, PRESETS.items()):
            col.button(label, width="stretch", key=f"preset_{label}",
                       on_click=apply_preset, args=(values,))

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
    bat_color, bowl_color = team_color(batting), team_color(bowling)

    with st.container(border=True):
        st.markdown("##### 🔮 Prediction")
        st.markdown(
            f"<div class='scoreboard'>"
            f"<div class='chip'><span class='lbl'>TARGET</span>"
            f"<span class='val'>{target:.0f}</span></div>"
            f"<div class='chip'><span class='lbl'>SCORE</span>"
            f"<span class='val' style='color:{bat_color}'>{score:.0f}/{wickets:.0f}</span></div>"
            f"<div class='chip'><span class='lbl'>OVERS</span>"
            f"<span class='val'>{overs:.0f}.{balls:.0f}</span></div>"
            f"<div class='chip' style='border-color:{bat_color}'>"
            f"<span class='lbl'>NEEDED</span>"
            f"<span class='val'>{result['runs_left']:.0f} off {result['balls_left']:.0f}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"### <span style='color:{bat_color}'>{batting}</span> win probability: "
            f"<span style='color:{bat_color}'>{p * 100:.1f}%</span>",
            unsafe_allow_html=True,
        )

        gcol, bcol = st.columns([1, 2], vertical_alignment="center")
        gcol.pyplot(draw_gauge(p, bat_color), width='content')
        with bcol:
            bat_seg = f"{team_short(batting)} {p * 100:.1f}%"
            bowl_seg = f"{team_short(bowling)} {(1 - p) * 100:.1f}%"
            st.markdown(
                f"<div class='winbar'>"
                f"<div class='seg' style='width:{max(p * 100, 14):.1f}%; "
                f"background:linear-gradient(90deg, {bat_color}, {bat_color}CC)'>{bat_seg}</div>"
                f"<div class='seg' style='width:{max((1 - p) * 100, 14):.1f}%; "
                f"background:linear-gradient(90deg, {bowl_color}CC, {bowl_color})'>{bowl_seg}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            crr, rrr = result["crr"], result["rrr"]
            if result["runs_left"] == 0:
                state_color, verdict = "#3FB68B", "The target is reached - chase complete."
            elif crr >= rrr + 1:
                state_color = "#3FB68B"
                verdict = f"{batting} are scoring comfortably faster than the required rate."
            elif crr <= rrr - 1:
                state_color = "#E45756"
                verdict = f"{batting} are falling behind the required rate."
            else:
                state_color = "#F2C94C"
                verdict = "Knife's edge - the scoring rate almost exactly matches the requirement."
            st.markdown(
                f"<div class='verdict' style='border-left-color:{state_color}'>"
                f"<b>{result['runs_left']:.0f}</b> needed off <b>{result['balls_left']:.0f}</b> balls "
                f"with <b>{10 - int(wickets)} wickets in hand</b>. {verdict}"
                f"</div>",
                unsafe_allow_html=True,
            )

        m_crr, m_rrr, m_runs, m_balls = st.columns(4)
        m_crr.metric("Current run rate", f"{result['crr']:.2f}")
        m_rrr.metric("Required run rate", f"{result['rrr']:.2f}",
                     delta=f"{crr - rrr:+.2f} vs current",
                     delta_color="normal" if crr >= rrr else "inverse",
                     help="Green when the batting team is scoring faster than required.")
        m_runs.metric("Runs left", f"{result['runs_left']:.0f}")
        m_balls.metric("Balls left", f"{result['balls_left']:.0f}")

    with st.expander("📈 Chase outlook", expanded=True):
        outlook, steps = chase_outlook(result, int(target), int(score), int(overs),
                                       int(balls), int(wickets), batting, bowling, venue)
        style = {
            "keep current rate": bat_color,
            "score at required rate": "#8AB4F8",
            "2 rpo slower than needed": "#E45756",
        }
        fig_o, ax_o = plt.subplots(figsize=(7.2, 3.2), dpi=150)
        for label, part in outlook.groupby("scenario"):
            ax_o.plot(part["balls from now"], part["p"], marker="o", markersize=3.5,
                      linewidth=1.8, label=label, color=style.get(label, "#888"))
        ax_o.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
        ax_o.set_xlabel("balls from now")
        ax_o.set_ylabel("P(chasing team wins)")
        ax_o.set_ylim(-0.02, 1.05)
        ax_o.set_title(f"If the next {steps} over(s) go like ...", fontsize=10)
        ax_o.legend(loc="lower left", fontsize=8)
        fig_o.tight_layout()
        st.pyplot(fig_o)
        st.caption("Projection assumes **no further wickets** and steady scoring at each "
                   "scenario's rate; the required-rate path completes the chase by definition.")

    with st.expander("What drives this prediction"):
        importance = get_importance()
        st.markdown(
            "The model weighs, in order: "
            + ", ".join(f"**{k}** ({v:.0%})" for k, v in importance.items())
            + f". For this situation the key numbers are a required rate of "
            f"**{rrr:.2f}** against a current rate of **{crr:.2f}**, with "
            f"**{10 - int(wickets)} wickets in hand** and "
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
                [{
                    "scenario": name,
                    "win probability": f"{prob * 100:.1f}%",
                    "change": f"{'▲' if prob >= p else '▼'} {abs(prob - p) * 100:.1f} pts",
                } for name, prob in rows]
            )
            st.table(what_if.set_index("scenario"))
        st.caption("Scenario probabilities move the situation and re-predict; "
                   "they assume the same venue and teams.")

    with st.expander("📋 Copy a match summary"):
        summary_text = (
            f"{batting} {score:.0f}/{wickets:.0f} ({overs:.0f}.{balls:.0f} ov) "
            f"chasing {target:.0f} at {venue}\n"
            f"CRR {result['crr']:.2f} | RRR {result['rrr']:.2f} | "
            f"win probability {p * 100:.1f}%"
        )
        st.code(summary_text, language=None)
        st.caption("Paste this anywhere - the numbers are the model's inputs and output.")

# ---------------------------------------------------------------------------
# Match replay
# ---------------------------------------------------------------------------

with replay_tab:
    st.markdown("Win probability through real chases from the test seasons (2023-24).")
    meta = REPLAY.drop_duplicates("match_id").set_index("match_id")

    def match_label(match_id: int) -> str:
        row = meta.loc[match_id]
        return (f"{int(row['season_year'])}: {row['batting_team']} chasing "
                f"{int(row['target'])} vs {row['bowling_team']} · {row['venue'].split(', ')[-1]}")

    match_ids = sorted(meta.index, key=match_label)
    choice = st.selectbox("Pick a match", match_ids, format_func=match_label, key="replay_match")
    demo = REPLAY[REPLAY["match_id"] == choice].sort_values(["over", "ball"])
    st.image(replay_curve_png(int(choice)), width='stretch')

    outcome = "chase won" if demo["chase_won"].iloc[-1] == 1 else "chase lost"
    st.caption(
        f"Finished **{demo['runs_so_far'].iloc[-1]:.0f}/{demo['wickets_out'].iloc[-1]:.0f}** "
        f"- **{outcome}**. Red crosses mark wickets; the dashed line is 50%. "
        "Each point uses only information available at that ball."
    )

# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

with about_tab:
    meta_app = get_meta()
    fin_app = meta_app["results"]["final_model"]
    split_app = meta_app["split"]
    st.markdown("#### How it works")
    st.markdown(
        "A regularised logistic regression on ball-by-ball 2nd-innings states. "
        "Features use **only information available at each ball**: target, runs "
        "and balls left, wickets in hand, current and required run rates, and "
        "each team's rolling win rate over strictly earlier matches. Matches are "
        "split **by season** (never by row), so no match leaks across splits; "
        "ties and no-result matches are excluded."
    )
    st.markdown("#### Measured performance (test seasons 2023-24)")
    overall = pd.DataFrame([
        {"split": "train 2008-21", "matches": split_app["train_matches"], "balls": split_app["train_balls"]},
        {"split": "validation 2022", "matches": split_app["validation_matches"], "balls": split_app["validation_balls"]},
        {"split": "test 2023-24", "matches": split_app["test_matches"], "balls": split_app["test_balls"]},
    ]).set_index("split")
    st.table(overall)
    test_f = fin_app["test_uncalibrated"]
    stage_rows = [{"stage": s["stage"], "log loss": round(s["log_loss"], 4),
                   "Brier": round(s["brier"], 4), "ROC-AUC": round(s["roc_auc"], 4)}
                  for s in meta_app["results"]["stage_metrics_test"]]
    stage_rows.insert(0, {"stage": "ALL BALLS", "log loss": round(test_f["log_loss"], 4),
                          "Brier": round(test_f["brier"], 4), "ROC-AUC": round(test_f["roc_auc"], 4)})
    st.table(pd.DataFrame(stage_rows).set_index("stage"))
    st.markdown("#### Top drivers")
    top_importance = dict(list(meta_app["feature_importance"].items())[:6])
    st.markdown(", ".join(f"**{k}** ({v:.0%})" for k, v in top_importance.items()))
    st.markdown("#### Limitations")
    st.markdown(
        "- No player-level strength, weather, dew or pitch data; venue is a coarse proxy.\n"
        "- Rolling team win rates ignore opponent and venue adjustment.\n"
        "- The 2023-24 impact-player era shifted scoring: at the low end the model is\n"
        "  underconfident about the current era (chases with ~3% predicted chance won ~10%).\n"
        "- Chases are assumed to be 120 balls (20 overs)."
    )
    st.warning("**Not for betting or gambling** - a learning and portfolio project.",
               icon="⚠️")

st.divider()
st.caption(
    "Model: logistic regression (season split 2008-21 / 2022 / 2023-24) · "
    "Data: Kaggle IPL matches & deliveries · Built for learning and portfolio use - "
    "**not for betting or gambling**."
)
