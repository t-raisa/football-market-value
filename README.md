# Forecasting Football Player Market Values from Public Transfermarkt Data

A time-aware, explainable machine-learning study. For each player at a valuation date *t*, we predict the Transfermarkt market value about **12 months later**, using **only information available at *t***.

Everything here is built from a free public dataset. One script checks the data version and rebuilds every table and figure from the raw download.

## What is in this repository

| Folder / file | Contents |
|---|---|
| `notebooks/` | `01_setup_data` → `02_features` → `03_models` → `04_analysis` → `05_robustness` → `06_review_checks` (run in this order) |
| `src/` | Shared code (see the list below) |
| `results/` | All result tables (CSV), `best_params.json`, `best_simple_baseline.json`, `note_dropped_rows.md` |
| `figures/` | All figures (PNG, 300 dpi) |
| `reference_results/` | Copy of `results/` used by the reproducibility check |
| `data/raw_checksums.sha256` | SHA-256 of the dataset version used in the paper (the zip and every CSV inside it) |
| `run_all.sh` | Checks the data version, runs all notebooks, compares all tables with the reference, and builds the paper |
| `scripts/` | `verify_data.py` (data version check), `check_results.py` (table comparison), `make_paper_assets.py` (paper numbers and tables) |
| `paper/` | IEEE conference draft (`main.tex`) and supplement; their numbers and tables are generated from `results/` |
| `environment.yml` | Conda environment with pinned versions |
| `environment.lock.yml` | Full lock file (every package) for Apple Silicon (osx-arm64) |

The modules in `src/`:
- `config.py`: paths, seed, split dates, analysis bins;
- `features.py`: leak-free features;
- `data.py`: splits;
- `baselines.py`: simple baselines (age × value curve, momentum);
- `models.py`: Ridge, tree models, GRU;
- `metrics.py`;
- `analysis.py`: player-level bootstrap;
- `robustness.py`: rolling-origin folds, ablation.

Not in git (re-created by the notebooks): the raw zip, `data/processed/*.parquet`, `models/`, and the compiled PDFs.

## 1. Get the data

1. Download the Kaggle dataset **"Football Data from Transfermarkt"** by davidcariboo:
   <https://www.kaggle.com/datasets/davidcariboo/player-scores>
2. Save the downloaded archive as `data/raw/transfermarkt_davidcariboo.zip`. **Do not unzip it:** notebook 01 reads the CSVs straight from the zip.
3. Please check the dataset's license on its Kaggle page before redistributing any data.

### Getting the same dataset version

The dataset is updated regularly. A newer download gives slightly different numbers, because new valuations, matches and transfers are added and earlier rows can be corrected. The paper uses the version whose CSV files are dated **5 September 2026**.

| | |
|---|---|
| CSV dates inside the zip | 2026-09-05 |
| zip size | 245,743,828 bytes |
| zip SHA-256 | `5971ec002b9d7b3e99358ed28b0518e7f384549db53075efc273cc57f7dacdf1` |

- **Where to find it:** Kaggle keeps earlier versions of a dataset. On the dataset page, choose the version published around 5 September 2026 and download that one.
- **Check your copy:** run the command below. It compares the zip and each CSV inside it with `data/raw_checksums.sha256`. Ten CSVs are used; two (`game_lineups.csv`, `game_events.csv`) are in the zip but not used.

  ```bash
  ~/miniforge3/envs/football/bin/python scripts/verify_data.py
  ```

- **If that version is no longer available:** the pipeline still runs on a newer version. Start it with `ALLOW_OTHER_DATA=1 ./run_all.sh`. Expect small differences from the paper's numbers, and the table check will report them.
- **Tip:** keep your own copy of the zip you used. Its checksum identifies the version exactly.

## 2. Set up the environment

The code was run with [Miniforge](https://github.com/conda-forge/miniforge) (conda-forge) on an Apple M2 (8 GB RAM, macOS).

```bash
conda env create -f environment.yml
```

```bash
conda run -n football python -m ipykernel install --user --name football --display-name "Python 3.11 (football)"
```

- On Apple Silicon, `environment.lock.yml` reproduces the exact environment, including every dependency.
- The GRU uses the Apple GPU through PyTorch's `mps` device and falls back to the CPU elsewhere. No TensorFlow is needed.
- `tectonic` (a small LaTeX engine) compiles the paper; it downloads the LaTeX packages it needs on first use.

## 3. Reproduce everything

```bash
./run_all.sh
```

This does four things, in order:
1. checks that the zip is the paper's dataset version (it stops if not, unless `ALLOW_OTHER_DATA=1`);
2. runs notebooks 01–06, saving their outputs in place;
3. compares every table in `results/` with `reference_results/` (numbers must agree to 1e-6; columns that only record run time are ignored);
4. regenerates the paper's numbers and tables and compiles `paper/main.pdf` and `paper/supplement.pdf`.

If your conda environment is not in `~/miniforge3/envs/football`, set `FOOTBALL_ENV_BIN` to its `bin` folder first. To only run the table comparison:

```bash
./run_all.sh --check
```

**Run time on an Apple M2 (8 GB):** about 16 minutes in total, on an otherwise idle machine. It runs comfortably on an 8 GB machine.

| notebook | what it does | time |
|---|---|---|
| 01_setup_data | CSV (inside the zip) → Parquet with small data types | 13 s |
| 02_features | dataset, 50 leak-free features, splits, leakage tests | 27 s |
| 03_models | split comparison, simple baselines, tuning, all models, GRU, extra test sets | 11 min |
| 04_analysis | bootstrap CIs, error by group, SHAP, "undervalued" check, ranking vs age rules | 15 s |
| 05_robustness | 6 test years (rolling origin), feature-group and club-spending ablation | 2.5 min |
| 06_review_checks | dropped rows, forecast intervals + conformal calibration, weak-spot checks | 1.5 min |

**Reproducibility notes**
- The random seed is 42 everywhere (`src/config.py`). LightGBM runs in deterministic mode.
- A clean re-run on the original machine reproduced all tables exactly.
- The GRU is trained on the Apple GPU, which is not guaranteed to be bit-for-bit identical across machines or PyTorch versions. GRU rows may therefore differ slightly elsewhere; every other model should match.

## 4. Study design in brief

- **Rows:** one row per (player, valuation date *t*), from July 2014.
- **Target:** log of the valuation closest to *t* + 365 days (± 60 days). Models predict the **12-month log change**.
- **No leakage.** Every feature uses only records dated ≤ *t*, which is checked by a "cut the future" test in notebook 02. Snapshot columns are not used:
  - `contract_expiration_date`, `international_caps` and the current club/value columns in `players.csv`;
  - the club/league columns in `player_valuations.csv`, which for players without an earlier transfer hold the 2026 club.

  Instead, "club at *t*" is rebuilt from transfers and matches.
- **Leagues:** the main experiments use the **14 leagues** with match data for the whole period (first match before 2013). The other 17 leagues (data only from 2024/25) form an extra test set.
- **Time split:**
  - train: valuation date before July 2023, with target before July 2024;
  - validation: July 2023 – June 2024;
  - test: from July 2024.
- **Baselines:**
  - naive "no change";
  - age × value curve: the average change by age group × value quintile. It is the best simple baseline, chosen on validation;
  - age × value × league curve;
  - momentum (full and damped).
- **Models:** Ridge, Random Forest, XGBoost, LightGBM, CatBoost, and a small GRU over the last 10 valuations.
- **Forecast intervals:** quantile LightGBM (10/50/90%), with conformal calibration by age group. The calibration rows' targets all lie before the test period.
- **Uncertainty:** all confidence intervals come from a **player-level bootstrap** (1,000 draws).

## 5. Where each result is

| Topic | Tables (`results/`) | Figures (`figures/`) |
|---|---|---|
| Raw data, leakage evidence, league coverage | table01–table03 | fig01 |
| Dataset, splits, features, leakage test | table04–table06, table10 | fig02 |
| Random vs time split | table08, table09, table18, table18b | fig04 |
| Main test results (incl. simple baselines) | table07, table11, table11b, table12, table13, tableA1 | fig03, fig05, fig06 |
| Extra test sets, unknown-club rows | table14, table15 | – |
| Bootstrap CIs, paired differences, vs best simple baseline | table16, table17, table17b | fig06 |
| Error by group and league (vs no change and vs the curve) | table19, table20 | fig07, fig08 |
| SHAP | table21, table22, table22b | fig09–fig11 |
| "Undervalued" check, ranking vs age rules | table23, table23b | fig12 |
| Several test years (rolling origin) | table24, table25, table25b, table25c | fig13 |
| Feature-group and club-spending ablation | table26, table26b, table26c | fig14 |
| Dropped rows (selection), full-window sensitivity | table27–table31, `note_dropped_rows.md` | – |
| Forecast intervals and conformal calibration | table32, table32b, table33 | fig15 |
| New-league feature gap, snapshot features, real fees | table34–table37 | – |

## Citation

*Add the paper's reference here once it is published.*
