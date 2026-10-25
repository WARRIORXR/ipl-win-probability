# IPL Win Probability 🏏

An end-to-end data analysis and machine-learning project on the Indian Premier
League: what actually drives match outcomes, and a model that estimates the
batting team's **win probability at any ball of a chase** — served through an
interactive Streamlit app with a match-replay view of real 2023-24 chases.

> **Live app:** [PLACEHOLDER - add your Streamlit Community Cloud URL after deploying]
>
> **Disclaimer:** this project is for learning and portfolio demonstration
> only. It is **not for betting or gambling**. Predictions are statistical
> estimates from historical data, not advice, and the app states this.

## Problem statement

At any moment of a T20 chase a spectator can ask: *what is the chance the
batting team wins from here?* Answering it well requires (a) understanding
which factors historically drive IPL outcomes — toss, venue, team strength,
and (b) a model that consumes only information available at that instant of
the chase. This project builds both, with strict prevention of data leakage.

## Dataset

Kaggle IPL dataset — `matches.csv` (1,095 matches) and `deliveries.csv`
(260,920 ball-by-ball rows), seasons **2008-2024**. Place both files in
`data/raw/` (gitignored; see [Data setup](#data-setup)).

Notable data problems found and fixed in `src/data.py` (all rules are explicit
mapping dictionaries):

- 19 team-name variants → 14 canonical franchises (renames such as Delhi
  Daredevils → Delhi Capitals; RCB Bangalore → Bengaluru; Deccan Chargers
  merged into Sunrisers Hyderabad as the same franchise slot).
- 58 raw venue spellings → 36 canonical grounds (missing city suffixes,
  punctuation drift, renamed grounds like Feroz Shah Kotla → Arun Jaitley
  Stadium).
- Season span labels (`2007/08`, `2009/10`, `2020/21`) normalised to the year
  the season was played, validated against match dates.
- The `ball` column counts every delivery row (wides included), so legal-ball
  counts are recomputed from extras types.
- `is_wicket` wrongly counts 15 "retired hurt" rows as outs; a corrected
  `is_out` flag excludes them.
- `target_runs` is the DLS-adjusted target for all 21 rain-affected chases
  (plus 9 shortened chases without a D/L flag) — the model uses this actual
  target, which is known before the chase starts (no leakage).
- Ties (14, all decided by super over) and no-result matches (5) are excluded
  from the modelling set so the label means "the chase succeeded in
  regulation play". This leaves **1,076 matches / 123,976 ball-level rows**.

## Method

**Features** (`src/features.py`), computed at every ball of every 2nd innings
using only past information: target, runs/balls/wickets left, current and
required run rates (clipped to the physical maximum of 36.0), wickets in
hand, and each team's rolling win rate over *strictly prior* matches.
"Last-5-over" momentum features are built too — an ablation showed they add
nothing (validation log loss delta −0.0006), so they are **not deployed**.

**Validation design** — split strictly by season, never by row:

| Split | Seasons | Matches | Ball rows |
|---|---|---|---|
| Train | 2008-2021 | 858 | 98,452 |
| Validation (tuning + calibration gate) | 2022 | 74 | 8,693 |
| Test (reported metrics only) | 2023-24 | 144 | 16,831 |

No match can appear in two splits (`season_split` raises if one ever does),
and unit tests prove a mutated future ball never changes earlier features.

**Models**: a naive required-rate threshold baseline, regularised logistic
regression, random forest and XGBoost — all tuned on the validation season
only. Logistic regression (C = 0.03) won the validation comparison and is
deployed. Isotonic and Platt calibration were fitted on out-of-fold training
predictions and **rejected** by a paired statistical gate (neither improved
validation log loss beyond 2 standard errors); an isotonic fit on a single
validation season was measurably harmful (test log loss 0.482 → 0.763), which
is why the gate exists.

## Results (test seasons 2023-24, 144 matches — real measured numbers)

| Model | Val log loss | Test log loss | Test Brier | Test ROC-AUC |
|---|---|---|---|---|
| Naive: "win if RRR ≤ 9.0" | 1.0752 | 1.1139 * | 0.2355 | 0.7484 |
| Logistic regression (deployed) | **0.4472** | **0.4823** | **0.1557** | **0.8637** |
| Random forest (depth 16, leaf 50) | 0.4526 | – | – | – |
| XGBoost (depth 4, lr 0.05) | 0.4731 | – | – | – |

\* hard 0/1 probabilities clipped to [0.01, 0.99] just to keep log loss
finite — judge the baseline on ROC-AUC/accuracy.

**By stage of the chase** (deployed model, test set) — predictions improve as
the match progresses:

| Stage | Log loss | Brier | ROC-AUC |
|---|---|---|---|
| Overs 1-6 | 0.6157 | 0.2044 | 0.7883 |
| Overs 7-15 | 0.4798 | 0.1535 | 0.8605 |
| Overs 16-20 | 0.2849 | 0.0862 | 0.9517 |

**Top drivers** (normalized importance): venue (50%), batting team (18%),
bowling team (13%), required run rate (6%), runs left (3%). In plain English:
where you play matters most, who you are matters next, and the live
run-rate/wicket situation is what moves the number ball by ball.

**Honest calibration note**: the model is well calibrated from ~0.45 upward,
but at the low end it is underconfident about the current era — chases with a
predicted ~3% chance actually won ~10% of the time in 2023-24. The
impact-player rule (2023+) made high chases easier than anything in the
2008-21 training data; this is documented as a limitation, not hidden.

## EDA highlights (`notebooks/01_eda.ipynb`, all numbers printed from data)

1. The toss is close to irrelevant: toss winners won **50.9%** of 1,076
   completed matches (95% CI 47.9-53.9%).
2. Venues differ hugely for chasing: chases succeed **64.9%** in Jaipur vs
   **43.4%** at Chepauk (54.3% overall).
3. Scoring rose **161.0 → 189.6** mean first-innings total from 2008 to 2024,
   with the biggest jump in the 2023-24 impact-player era.
4. Leaders: V Kohli **8,014 runs**; AD Russell best strike rate **176.8**
   (min 500 balls); YS Chahal most bowler-credited wickets **205**; A Kumble
   best economy **6.58** (min 50 overs).
5. The same required rate means different things at different stages: at
   RRR ≈ 11, chases succeed **22.2%** of the time in overs 1-6 but **56.0%**
   in overs 16-20.

## The app

`app/app.py` (Streamlit): enter batting/bowling team, venue, target, current
score, overs + balls completed and wickets lost; see win probability for both
teams, CRR/RRR/runs/balls left, a "what drives this prediction" section with
what-if scenarios, and a **Match replay** tab that plots the win-probability
curve through any 2023-24 chase with wicket markers. Input validation shows
friendly errors (score beyond target, impossible over/ball counts, same
teams). Model loading is cached; only committed assets are used at runtime
(no raw data needed). A visible disclaimer states it is not for betting.

Screenshots (captured live, stored in `reports/screenshots/`):

| File | Shows |
|---|---|
| `app_live.png` | Dark-themed inputs with tooltips, form/venue context, one-click presets |
| `app_stats.png` | Prediction dashboard: scoreboard chips, team-coloured gauge, two-tone win bar, verdict |
| `app_outlook.png` | Chase outlook: win probability over the next overs under three scoring-rate scenarios |
| `app_drivers.png` | "What drives this prediction" + what-if table with point deltas |
| `app_thriller.png` | The "Final-over thriller" preset: 14 needed off 3 balls → 0.7% |
| `app_replay.png` | Match replay: team-coloured probability curve with wicket markers |
| `app_model_card.png` | Model card tab: split design, measured metrics, drivers, limitations |

Other app features: a broadcast-style scoreboard strip (target / score / overs /
needed), a required-rate delta chip (green when the batting side is winning the
rate battle), venue context (historical chase win rate at the selected ground),
team rolling-form captions, and a copyable match summary block.

Re-capture after deployment if you restyle anything, and replace the live-app
placeholder at the top.

## Repository layout

```
data/raw/            raw Kaggle CSVs (gitignored - download yourself)
data/processed/      cleaned + feature CSVs written by src (gitignored)
notebooks/01_eda.ipynb              EDA, executed top-to-bottom
notebooks/02_features_and_model.ipynb
src/data.py          loading, cleaning, mapping dictionaries
src/features.py      ball-by-ball 2nd-innings feature builder
src/train.py         season split, tuning, calibration gate, artifacts
src/evaluate.py      metrics, calibration tables, figures
src/predict.py       load artifacts, predict a single situation
app/app.py           Streamlit app (+ committed replay asset in app/assets/)
models/model.joblib  trained pipeline (committed, needed by the app)
models/metadata.json split, metrics, feature importance, versions
tests/               35 pytest tests incl. Streamlit AppTest UI suite
reports/figures/     charts used by notebooks and README
reports/screenshots/ app screenshots
```

## Run locally

```bash
# Python 3.11+ (developed on 3.15.0)
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# data (one-time): download matches.csv + deliveries.csv from Kaggle (see below)
python -m src.data report            # data-quality report
python -m src.data clean             # write data/processed/*.csv
python -m src.features               # build chase_features.csv
python -m src.train                  # train + evaluate + save models/
pytest                               # 35 tests
streamlit run app/app.py             # the app
```

### Data setup

1. Create a free Kaggle account.
2. Download the IPL dataset's `matches.csv` and `deliveries.csv`
   (2008-2024 edition, 1,095 matches / 260,920 deliveries).
3. Place them at `data/raw/matches.csv` and `data/raw/deliveries.csv`.
   Everything under `data/` is reproducible or gitignored; the app itself
   needs only the committed `models/` and `app/assets/` files.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (it is already public at
   `WARRIORXR/ipl-win-probability`).
2. Go to <https://share.streamlit.io> → **New app** → select the repo,
   branch `main`, main file `app/app.py`.
3. Deploy. `requirements.txt` pins everything; no secrets, no data upload and
   no environment variables are needed because `models/model.joblib` and
   `app/assets/replay_chases.csv` are committed.
4. Copy the app URL into the placeholder at the top of this README and push.

## Limitations

- No player-level strength (form, match-ups), no weather, dew or pitch
  reports; venue is a coarse proxy for pitch conditions.
- Rolling team win rates ignore opponent and venue adjustment.
- The 2023-24 impact-player era shifted scoring; the model trained through
  2021 is underconfident about extreme chases in the test seasons.
- Chases are assumed to be 120 balls (20 overs); DLS-revised chases in the
  app's live tab should be entered via the equivalent rate, not raw balls.
- Ties and no-result matches are excluded by design.

## Performance (measured)

`python scripts/profile_app.py` (Python 3.15, dev machine): a single situation
prediction takes **3.3 ms** p50 / **4.4 ms** p95 over 200 runs; the entire
16,831-row test-season replay batch **23 ms**; one match's replay curve ~4 ms;
a matplotlib figure render ~30-40 ms. Inference is a regularised linear model
on scikit-learn's BLAS-backed numpy, so a native (Rust) rewrite of the
prediction maths would not be perceptible in the app - the time budget is
dominated by chart rendering and Streamlit's rerun cycle. That is where the
app optimises instead: the model artifact and venue stats are cached, the
chase-outlook projection predicts all scenario rows in **one vectorized batch
call**, and each replay match's rendered curve is cached as PNG bytes keyed by
match id, so repeat selections and tab switches are free.

## Tests and reproducibility

- 35 tests (`pytest`): cleaning rules, feature edge cases (wides, retired
  hurt, RRR when no balls remain), leakage proofs (mutating future balls
  never changes earlier features; the label flips without feature changes),
  split disjointness, artifact consistency, situation predictions, and a full
  Streamlit AppTest UI suite (rendering, validation errors, replay tab).
- Fixed seed 42 everywhere; `models/metadata.json` records library versions.
- Both notebooks execute top-to-bottom without errors (nbconvert-verified).

## Resume bullets (drafts - only real measured numbers)

- Built an end-to-end IPL win-probability system on 260,920 ball-by-ball
  deliveries (2008-2024): engineered a leakage-free ball-level dataset of
  123,976 rows with documented cleaning of 19 team-name and 58 venue-name
  variants, validated by 35 automated tests including future-leakage proofs.
- Trained and time-aware-validated (train 2008-21 / validate 2022 / test
  2023-24, matches never split across sets) logistic regression, random
  forest and XGBoost models; deployed the best (test ROC-AUC **0.864**,
  log loss **0.482**, Brier **0.156**), beating a required-rate baseline
  (ROC-AUC 0.748), with predictions improving from AUC 0.788 (overs 1-6) to
  0.952 (overs 16-20).
- Shipped an interactive Streamlit app (live win probability, what-if
  analysis, match-replay curves) with cached model loading, input validation
  and a [PLACEHOLDER - live app URL]; also rejected single-season isotonic
  calibration via a statistical gate after it degraded test log loss from
  0.482 to 0.763.

## License

MIT — see [LICENSE](LICENSE). Again: **not for betting or gambling**.
