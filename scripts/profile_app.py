"""Measured latency profile of the deployed model and app components.

Run from the repository root:  python scripts/profile_app.py

Prints the numbers quoted in the README's performance section. The point of
this script is honesty about speed: the deployed model is a linear model on
scikit-learn's BLAS-backed numpy, so inference is already ~3 ms and the app's
perceived speed is governed by chart rendering and Streamlit's rerun cycle,
not by the prediction maths.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.predict import load_model, predict_situation  # noqa: E402


def main() -> None:
    """Time artifact loading, predictions and figure rendering."""
    t0 = time.perf_counter()
    art = load_model()
    print(f"artifact load (cold):        {(time.perf_counter() - t0) * 1000:8.1f} ms")

    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        predict_situation("Chennai Super Kings", "Mumbai Indians",
                          "Wankhede Stadium, Mumbai", 180, 90, 10, 3, 3)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"predict_situation p50/p95:   {times[100]:8.2f} / {times[190]:.2f} ms")

    replay = pd.read_csv(ROOT / "app" / "assets" / "replay_chases.csv")
    t0 = time.perf_counter()
    art["pipeline"].predict_proba(replay[art["feature_columns"]])
    print(f"batch predict {len(replay):,} rows: {(time.perf_counter() - t0) * 1000:8.1f} ms")

    demo = replay[replay["match_id"] == replay["match_id"].iloc[0]]
    t0 = time.perf_counter()
    art["pipeline"].predict_proba(demo[art["feature_columns"]])[:, 1]
    print(f"one replay match ({len(demo)} balls): {(time.perf_counter() - t0) * 1000:8.1f} ms")

    probs = art["pipeline"].predict_proba(demo[art["feature_columns"]])[:, 1]
    renders = []
    for _ in range(5):
        t0 = time.perf_counter()
        fig = ev.plot_win_curve(demo["balls_bowled"].tolist(), probs.tolist(), [30, 60])
        renders.append((time.perf_counter() - t0) * 1000)
        import matplotlib.pyplot as plt

        plt.close(fig)
    renders.sort()
    print(f"win-curve figure render med: {renders[2]:8.1f} ms")


if __name__ == "__main__":
    main()
