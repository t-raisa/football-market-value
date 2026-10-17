# Forecasting Football Player Market Values from Public Transfermarkt Data

A time-aware, explainable machine-learning study. For each player at a valuation date *t*, we predict the Transfermarkt market value about **12 months later**, using **only information available at *t***.

Everything here is built from a free public dataset. One script rebuilds every table and figure from the raw download.

## What is in this repository

| Folder / file | Contents |
|---|---|
| `notebooks/` | `01_setup_data` → `02_features` → `03_models` → `04_analysis` → `05_robustness` (run in this order) |
| `src/` | Shared code: `config.py` (paths, seed, split dates, bins), `features.py` (leak-free features), `data.py` (splits), `models.py` (baselines, tree models, GRU), `metrics.py`, `analysis.py` (player-level bootstrap), `robustness.py` |
| `results/` | All result tables (CSV) and `best_params.json` |
| `figures/` | All figures (PNG, 300 dpi) |
| `reference_results/` | Copy of `results/` used by the reproducibility check |
| `run_all.sh`, `scripts/check_results.py` | Rebuild everything and compare with the reference |
| `environment.yml` | Conda environment with pinned versions |
| `environment.lock.yml` | Full lock file (every package) for Apple Silicon (osx-arm64) |

Not in git (re-created by the notebooks): the raw zip, `data/processed/*.parquet`, and `models/`.

## 1. Get the data

1. Download the Kaggle dataset **"Football Data from Transfermarkt"** by davidcariboo:
   <https://www.kaggle.com/datasets/davidcariboo/player-scores>
2. Save the downloaded archive as `data/raw/transfermarkt_davidcariboo.zip`. **Do not unzip it:** notebook 01 reads the CSVs straight from the zip.
3. Please check the dataset's license on its Kaggle page before redistributing any data.

**Data version used in the paper.** The dataset is updated regularly, so a newer download will give slightly different numbers. The version we used:

| | |
|---|---|
| CSV dates inside the zip | 2026-09-05 |
| zip size | 245,743,828 bytes |
| SHA-256 | `5971ec002b9d7b3e99358ed28b0518e7f384549db53075efc273cc57f7dacdf1` |

Check your copy with:

```bash
shasum -a 256 data/raw/transfermarkt_davidcariboo.zip
```

Used tables: players, player_valuations, appearances, transfers, clubs, competitions, games, club_games, countries, national_teams. `game_lineups.csv` and `game_events.csv` are not used.

## 2. Set up the environment

The code was run with [Miniforge](https://github.com/conda-forge/miniforge) (conda-forge) on an Apple M2 (8 GB RAM, macOS).

```bash
conda env create -f environment.yml
```

```bash
conda run -n football python -m ipykernel install --user --name football --display-name "Python 3.11 (football)"
```

On Apple Silicon, `environment.lock.yml` reproduces the exact environment, including every dependency. The GRU uses the Apple GPU through PyTorch's `mps` device and falls back to the CPU elsewhere. No TensorFlow is needed.

## 3. Reproduce everything

```bash
./run_all.sh
```

This runs notebooks 01–05 in order and saves their outputs in place. It then compares every table in `results/` with `reference_results/`; numbers must agree to 1e-6. Columns that only record run time are ignored.

If your conda environment is not in `~/miniforge3/envs/football`, set `FOOTBALL_ENV_BIN` to its `bin` folder first. To only run the comparison:

```bash
./run_all.sh --check
```

**Run time on an Apple M2 (8 GB):** about 15 minutes in total. It runs comfortably on an 8 GB machine.

| notebook | what it does | time |
|---|---|---|
| 01_setup_data | CSV (inside the zip) → Parquet with small data types | 14 s |
| 02_features | dataset, 50 leak-free features, splits, leakage tests | 28 s |
| 03_models | split comparison, tuning, all models, GRU, extra test sets | 11.5 min |
| 04_analysis | bootstrap CIs, error by group, SHAP, "undervalued" check | 16 s |
| 05_robustness | 6 test years (rolling origin), feature-group ablation | 2.5 min |

**Reproducibility notes**
- The random seed is 42 everywhere (`src/config.py`). LightGBM runs in deterministic mode.
- A clean re-run on the original machine reproduced all tables exactly.
- The GRU is trained on the Apple GPU, which is not guaranteed to be bit-for-bit identical across machines or PyTorch versions. GRU rows may therefore differ slightly elsewhere; every other model should match.

## 4. Study design in brief

- **Rows:** one row per (player, valuation date *t*), from July 2014.
- **Target:** log of the valuation closest to *t* + 365 days (± 60 days). Models predict the **12-month log change**.
- **No leakage.** Every feature uses only records dated ≤ *t*, which is checked by a "cut the future" test in notebook 02. Snapshot columns are not used:
  - `contract_expiration_date`, `international_caps` and the current club/value columns in `players.csv`.
  - The club/league columns in `player_valuations.csv`: for players without an earlier transfer, these hold the 2026 club.
  - Instead, "club at *t*" is rebuilt from transfers and matches.
- **Leagues:** the main experiments use the **14 leagues** with match data for the whole period (first match before 2013). The other 17 leagues (data only from 2024/25) form an extra test set.
- **Time split:**
  - train: valuation date before July 2023, with target before July 2024
  - validation: July 2023 – June 2024
  - test: from July 2024
- **Comparison splits:** a random 80/20 split by rows and by player is reported only to show how random splits over-state accuracy.
- **Models:** naive "no change", Ridge, Random Forest, XGBoost, LightGBM, CatBoost, and a small GRU over the last 10 valuations.
- **Uncertainty:** all confidence intervals come from a **player-level bootstrap** (1,000 draws).

## 5. Where each result is

| Topic | Tables (`results/`) | Figures (`figures/`) |
|---|---|---|
| Raw data, leakage evidence, league coverage | table01–table03 | fig01 |
| Dataset, splits, features, leakage test | table04–table06, table10 | fig02 |
| Random vs time split | table08, table09, table18, table18b | fig04 |
| Main test results | table07, table11–table13, tableA1 | fig03, fig05, fig06 |
| Extra test sets, unknown-club rows | table14, table15 | – |
| Bootstrap CIs and paired differences | table16, table17 | fig06 |
| Error by group and league | table19, table20 | fig07, fig08 |
| SHAP | table21, table22, table22b | fig09–fig11 |
| "Undervalued" check | table23 | fig12 |
| Several test years (rolling origin) | table24, table25, table25b | fig13 |
| Feature-group and club-spending ablation | table26, table26b, table26c | fig14 |

## Citation

*Add the paper's reference here once it is published.*
