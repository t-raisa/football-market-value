"""Robustness checks: rolling-origin folds (several test years) and feature-group ablation."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import SEED
from .features import FEATURES, FEATURE_INFO

# LightGBM settings fixed BEFORE any tuning (the untuned settings of notebook 03, part A3).
# Used for the rolling-origin folds so that no fold uses settings chosen on later data.
LGB_FIXED = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8)

GROUPS = list(dict.fromkeys(g for g, _ in FEATURE_INFO.values()))     # player, value history, ...


def group_features(group):
    return [f for f in FEATURES if FEATURE_INFO[f][0] == group]


def fold(main: pd.DataFrame, test_year: int, open_end=False) -> dict:
    """Frames for one rolling-origin fold, with the same rules as the main split:
      test       : t in [test_year-07-01, test_year+1-07-01)  (open_end: no upper limit)
      validation : the year before the test year
      train      : t before the validation year AND target date before the test year
      tune       : train rows whose target date is before the validation year (embargo)"""
    test_start = pd.Timestamp(f"{test_year}-07-01")
    val_start = pd.Timestamp(f"{test_year - 1}-07-01")
    test_end = pd.Timestamp(f"{test_year + 1}-07-01")
    train = main[(main.t < val_start) & (main.target_date < test_start)]
    in_test = main.t >= test_start if open_end else (main.t >= test_start) & (main.t < test_end)
    return {"tune": train[train.target_date < val_start].reset_index(drop=True),
            "train": train.reset_index(drop=True),
            "val": main[(main.t >= val_start) & (main.t < test_start)].reset_index(drop=True),
            "test": main[in_test].reset_index(drop=True)}


def change(d):
    return (d["y"] - d["log_value"]).astype("float64")


def fit_lgb(params, feats, tune, val, final, max_trees=5000):
    """Early stopping on validation (fit on the tuning rows) picks the number of trees;
    then refit on `final` with that many trees. Returns (model, n_trees)."""
    base = dict(random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1)
    es = lgb.LGBMRegressor(n_estimators=max_trees, **base, **params)
    es.fit(tune[feats], change(tune), eval_set=[(val[feats], change(val))], eval_metric="l1",
           callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
    n = int(es.best_iteration_)
    model = lgb.LGBMRegressor(n_estimators=n, **base, **params).fit(final[feats], change(final))
    return model, n


def predict_log(model, feats, d):
    return d["log_value"].to_numpy() + model.predict(d[feats])
