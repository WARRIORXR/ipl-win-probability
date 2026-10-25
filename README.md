# IPL Win Probability

End-to-end data analysis and machine-learning project on Indian Premier League
(IPL) cricket: what drives match outcomes, and a model that estimates the
batting team's win probability at any ball of a chase — served through an
interactive Streamlit app.

> **Status: Phase 0 (setup & data check) — work in progress.**

> **Disclaimer:** This project is for learning and portfolio demonstration
> only. It is **not** intended for betting, gambling, or wagering of any kind.
> Predictions are statistical estimates on historical data, not advice.

## Repository layout

```
data/raw/            raw Kaggle CSVs (gitignored - see download instructions)
data/processed/      cleaned CSVs written by src/data.py (gitignored)
notebooks/           01_eda.ipynb, 02_features_and_model.ipynb
src/                 data.py, features.py, train.py, evaluate.py, predict.py
app/app.py           Streamlit app
models/              saved model + metadata JSON (committed)
tests/               pytest suite
reports/figures/     charts used in notebooks/README
```

## Data

Kaggle IPL dataset — `matches.csv` and `deliveries.csv`, 2008-2024 edition
(1,095 matches, 260,920 deliveries). Download both files from Kaggle and place
them in `data/raw/`:

1. Create a free Kaggle account and accept the competition/dataset terms if asked.
2. Download the two CSVs (`matches.csv`, `deliveries.csv`).
3. Put them at `data/raw/matches.csv` and `data/raw/deliveries.csv`.

Nothing under `data/` is committed; `src/data.py` reproduces `data/processed/`
from the raw files with documented mapping dictionaries.

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash (.venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

python -m src.data report          # data-quality report
python -m src.data clean           # writes data/processed/*.csv
pytest                             # run tests
```

## Project plan

- **Phase 0** - setup, data check, cleaning rules (`src/data.py`)
- **Phase 1** - EDA (`notebooks/01_eda.ipynb`)
- **Phase 2** - ball-by-ball 2nd-innings features + tests (`src/features.py`)
- **Phase 3** - time-aware modelling (`src/train.py`, `src/evaluate.py`)
- **Phase 4** - Streamlit app (`app/app.py`)
- **Phase 5** - deployment, final README

## License

MIT — see [LICENSE](LICENSE).
