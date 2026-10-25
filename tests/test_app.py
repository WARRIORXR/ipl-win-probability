"""UI tests for app/app.py using Streamlit's official AppTest framework.

AppTest executes the app script like a real browser session: widgets can be
set, reruns triggered, and the rendered elements inspected - so validation
messages, the probability output and the replay chart are all checked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path("models/model.joblib").exists(),
    reason="models/model.joblib not trained yet (run python -m src.train)",
)


@pytest.fixture()
def at() -> AppTest:
    """A freshly run app with default inputs."""
    run = AppTest.from_file(APP, default_timeout=120)
    run.run()
    return run


def test_app_runs_without_exception(at: AppTest) -> None:
    assert not at.exception
    assert not at.error, [e.value for e in at.error]


def test_disclaimer_is_visible(at: AppTest) -> None:
    texts = " ".join(str(el.value) for el in list(at.warning) + list(at.markdown) + list(at.caption))
    assert "betting" in texts.lower() and "gambling" in texts.lower()


def test_default_prediction_rendered(at: AppTest) -> None:
    headers = " ".join(s.value for s in at.subheader)
    assert "win probability" in headers.lower()
    assert len(at.metric) == 4, "CRR, RRR, runs left and balls left must all render"
    assert len(at.progress) == 1


def test_prediction_changes_with_inputs(at: AppTest) -> None:
    before = at.subheader[0].value
    at.number_input(key="score").set_value(170)
    at.number_input(key="wickets").set_value(1)
    at.run()
    assert not at.exception
    assert at.subheader[0].value != before, "a big score with 1 wicket down must move the headline"
    assert "%" in at.subheader[0].value


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
    assert "Final outcome" in captions, "the replay caption with the match result must render"
    assert not at.exception
    # AppTest .options are formatted labels; .select() takes the raw match id
    import pandas as pd

    ids = pd.read_csv("app/assets/replay_chases.csv")["match_id"].unique()
    other = int([i for i in ids if i != select.value][0])
    select.select(other).run()
    assert not at.exception, "switching matches must not raise"
    assert at.selectbox(key="replay_match").value == other


def test_whatif_table_present(at: AppTest) -> None:
    assert len(at.table) >= 1, "the what-if scenario table must render"
