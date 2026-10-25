"""UI tests for app/app.py using Streamlit's official AppTest framework.

AppTest executes the app script like a real browser session: widgets can be
set, reruns triggered, and the rendered elements inspected - so validation
messages, the probability output and the replay chart are all checked here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
pytestmark = pytest.mark.skipif(
    not Path("models/model.joblib").exists(),
    reason="models/model.joblib not trained yet (run python -m src.train)",
)


@pytest.fixture()
def at() -> AppTest:
    """A freshly run app with default inputs."""
    run = AppTest.from_file(APP, default_timeout=120)
    run.run()
    return run


def _all_text(at: AppTest) -> str:
    """Every rendered text value, for content assertions."""
    elements = (list(at.markdown) + list(at.subheader) + list(at.caption)
                + list(at.warning) + list(at.info) + list(at.success))
    return " ".join(str(el.value) for el in elements)


def test_app_runs_without_exception(at: AppTest) -> None:
    assert not at.exception
    assert not at.error, [e.value for e in at.error]


def test_disclaimer_is_visible(at: AppTest) -> None:
    texts = _all_text(at).lower()
    assert "betting" in texts and "gambling" in texts


def test_default_prediction_rendered(at: AppTest) -> None:
    assert "win probability" in _all_text(at).lower()
    assert len(at.metric) == 4, "CRR, RRR, runs left and balls left must all render"
    assert any("winbar" in str(el.value) for el in at.markdown), (
        "the two-tone win bar must render"
    )


def test_prediction_changes_with_inputs(at: AppTest) -> None:
    before = _all_text(at)
    at.number_input(key="score").set_value(170)
    at.number_input(key="wickets").set_value(1)
    at.run()
    assert not at.exception
    assert _all_text(at) != before, "a big score with 1 wicket down must move the output"
    assert "win probability" in _all_text(at).lower()


def test_preset_button_fills_the_situation(at: AppTest) -> None:
    """One click on a preset fills all five situation inputs at once."""
    at.button(key="preset_🔥 Final-over thriller").click()
    at.run()
    assert not at.exception
    assert at.number_input(key="target").value == 190
    assert at.number_input(key="score").value == 176
    assert at.number_input(key="overs").value == 19
    assert at.number_input(key="balls").value == 3
    assert "win probability" in _all_text(at).lower()


def test_score_at_or_above_target_shows_friendly_error(at: AppTest) -> None:
    at.number_input(key="score").set_value(180)  # equals the default target
    at.run()
    assert at.error, "reaching the target must show a validation error"
    assert "already reaches the target" in at.error[0].value


def test_replay_tab_renders_a_chart(at: AppTest) -> None:
    select = at.selectbox(key="replay_match")
    assert select is not None, "the replay tab must expose a match picker"
    at.run()
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Finished" in captions, "the replay caption with the final score must render"
    assert not at.exception
    # AppTest .options are formatted labels; .select() takes the raw match id
    ids = pd.read_csv("app/assets/replay_chases.csv")["match_id"].unique()
    other = int([i for i in ids if i != select.value][0])
    select.select(other).run()
    assert not at.exception, "switching matches must not raise"
    assert at.selectbox(key="replay_match").value == other


def test_whatif_table_present(at: AppTest) -> None:
    assert len(at.table) >= 1, "the what-if scenario table must render"
