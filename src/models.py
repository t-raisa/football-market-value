"""Models for the 12-month log change: baselines, tree ensembles with light tuning, and a GRU.

Protocol (same for every model):
  1. Tuning: fit on train rows whose target date is before the validation period
     (`train_target_before_val`), score each setting on the validation split (MAE of the change).
  2. Final fit: the chosen setting is refit on ALL train rows (targets before the test period).
     Boosting models keep the number of trees found by early stopping in step 1.
  3. Validation rows are never used for the final fit: their targets fall in the test period.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from .config import SEED
from .data import xy
from .features import FEATURES, CATEGORICAL

NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


# ----------------------------------------------------------------------------- preprocessing
def _as_str(X):
    return X.astype("string").fillna("missing").astype(object)


def tabular_encoder(scale: bool):
    """Median-impute numbers (+ a 0/1 'was missing' column each), one-hot the categories.
    Used by Ridge, Random Forest and the GRU's dense branch."""
    num = [SimpleImputer(strategy="median", add_indicator=True)] + ([StandardScaler()] if scale else [])
    cat = make_pipeline(FunctionTransformer(_as_str),
                        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                      sparse_output=False))
    return ColumnTransformer([("num", make_pipeline(*num), NUMERIC), ("cat", cat, CATEGORICAL)])


def _cat_frame(X):
    """CatBoost wants categories as plain strings."""
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("string").fillna("missing").astype(str)
    return X


# ----------------------------------------------------------------------------- model specs
# Each spec: grid of settings + build(params) -> (fit(X, y, Xval, yval) -> fitted, n_trees)
def _sample(grid, n, seed=SEED):
    full = list(ParameterGrid(grid))
    if len(full) <= n:
        return full
    idx = np.random.default_rng(seed).choice(len(full), size=n, replace=False)
    return [full[i] for i in sorted(idx)]


class Ridge_:
    name = "Ridge regression"
    grid = {"alpha": [1, 10, 100, 1000, 10000, 100000]}
    n_settings = 6

    def fit(self, p, X, y, Xv=None, yv=None, n_trees=None):
        self.m = make_pipeline(tabular_encoder(scale=True), Ridge(alpha=p["alpha"])).fit(X, y)
        return None

    def predict(self, X):
        return self.m.predict(X)


class RF_:
    name = "Random Forest"
    grid = {"max_features": [0.3, 0.5, 0.7], "min_samples_leaf": [3, 10, 30]}
    n_settings = 9

    def fit(self, p, X, y, Xv=None, yv=None, n_trees=None):
        rf = RandomForestRegressor(n_estimators=300, max_samples=0.5, n_jobs=-1, random_state=SEED, **p)
        self.m = make_pipeline(tabular_encoder(scale=False), rf).fit(X, y)
        return None

    def predict(self, X):
        return self.m.predict(X)


class LGB_:
    name = "LightGBM"
    grid = {"num_leaves": [15, 31, 63, 127], "min_child_samples": [20, 50, 100, 200],
            "colsample_bytree": [0.5, 0.7, 0.9], "subsample": [0.7, 0.9], "reg_lambda": [0, 1, 10],
            "objective": ["l2", "l1"]}
    n_settings = 15

    def fit(self, p, X, y, Xv=None, yv=None, n_trees=None):
        base = dict(learning_rate=0.03, subsample_freq=1, random_state=SEED, deterministic=True,
                    force_row_wise=True, verbose=-1)
        if n_trees is None:
            self.m = lgb.LGBMRegressor(n_estimators=5000, **base, **p)
            self.m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="l1",
                       callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
            return self.m.best_iteration_
        self.m = lgb.LGBMRegressor(n_estimators=n_trees, **base, **p).fit(X, y)
        return n_trees

    def predict(self, X):
        return self.m.predict(X)


class XGB_:
    name = "XGBoost"
    grid = {"max_depth": [4, 6, 8, 10], "min_child_weight": [1, 5, 20, 50], "subsample": [0.7, 0.9],
            "colsample_bytree": [0.5, 0.7, 0.9], "reg_lambda": [1, 5, 20],
            "objective": ["reg:squarederror", "reg:absoluteerror"]}
    n_settings = 10

    def fit(self, p, X, y, Xv=None, yv=None, n_trees=None):
        base = dict(tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, learning_rate=0.03,
                    eval_metric="mae", random_state=SEED, n_jobs=-1)
        if n_trees is None:
            self.m = xgb.XGBRegressor(n_estimators=5000, early_stopping_rounds=200, **base, **p)
            self.m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
            return self.m.best_iteration + 1
        self.m = xgb.XGBRegressor(n_estimators=n_trees, **base, **p).fit(X, y, verbose=False)
        return n_trees

    def predict(self, X):
        return self.m.predict(X)


class CAT_:
    name = "CatBoost"
    grid = {"depth": [4, 6, 8], "l2_leaf_reg": [3, 10, 30], "loss_function": ["RMSE", "MAE"]}
    n_settings = 9

    def fit(self, p, X, y, Xv=None, yv=None, n_trees=None):
        base = dict(learning_rate=0.06, random_seed=SEED, verbose=0, thread_count=-1,
                    cat_features=CATEGORICAL, eval_metric="MAE", allow_writing_files=False)
        if n_trees is None:
            self.m = cb.CatBoostRegressor(iterations=4000, od_type="Iter", od_wait=200,
                                          use_best_model=True, **base, **p)
            self.m.fit(_cat_frame(X), y, eval_set=(_cat_frame(Xv), yv))
            return self.m.get_best_iteration() + 1
        self.m = cb.CatBoostRegressor(iterations=n_trees, **base, **p).fit(_cat_frame(X), y)
        return n_trees

    def predict(self, X):
        return self.m.predict(_cat_frame(X))


MODELS = [Ridge_, RF_, LGB_, XGB_, CAT_]


# ----------------------------------------------------------------------------- tuning + final fit
def tune_and_fit(spec, tune_df, val_df, final_df, log=print):
    """Light tuning on validation, then refit the best setting on final_df.
    Returns (fitted model, best params, n_trees, tuning log DataFrame)."""
    Xt, yt = xy(tune_df)
    Xv, yv = xy(val_df)
    rows = []
    for p in _sample(spec.grid, spec.n_settings):
        t0 = time.time()
        m = spec()
        n_trees = m.fit(p, Xt, yt, Xv, yv)
        mae = float(np.mean(np.abs(m.predict(Xv) - yv)))
        rows.append({"model": spec.name, **{k: str(v) for k, v in p.items()}, "n_trees": n_trees,
                     "val_MAE_change": mae, "seconds": round(time.time() - t0, 1)})
        log(f"  {spec.name:16s} {p}  trees={n_trees}  val MAE={mae:.4f}  ({time.time() - t0:.0f}s)")
    tlog = pd.DataFrame(rows)
    best = tlog["val_MAE_change"].idxmin()
    best_p = _sample(spec.grid, spec.n_settings)[best]
    best_trees = tlog.loc[best, "n_trees"]
    best_trees = None if pd.isna(best_trees) else int(best_trees)
    Xf, yf = xy(final_df)
    final = spec()
    final.fit(best_p, Xf, yf, n_trees=best_trees)
    return final, best_p, best_trees, tlog


# ----------------------------------------------------------------------------- GRU
SEQ_LEN = 10


def build_sequences(df: pd.DataFrame, vals: pd.DataFrame, seq_len=SEQ_LEN) -> np.ndarray:
    """For each row (player, t): the player's last `seq_len` valuations on/before t (t included),
    oldest first, left-padded. Per step: [log value - log value at t, years before t, is_real].
    Only valuations dated <= t are read, so the sequence is leak-free."""
    v = vals[vals["market_value_in_eur"] > 0].sort_values(["player_id", "date"], kind="stable")
    pid = v["player_id"].to_numpy()
    date = v["date"].to_numpy().astype("datetime64[D]")
    logv = np.log(v["market_value_in_eur"].to_numpy(dtype="float64"))
    pos = pd.Series(np.arange(len(v)), index=pd.MultiIndex.from_arrays([pid, date]))
    key = pd.MultiIndex.from_arrays([df["player_id"].to_numpy(), df["t"].to_numpy().astype("datetime64[D]")])
    idx = pos.reindex(key).to_numpy()
    assert not np.isnan(idx).any(), "every dataset row must be a valuation"
    idx = idx.astype(int)
    offs = np.arange(-seq_len + 1, 1)                       # -9 ... 0
    gather = idx[:, None] + offs[None, :]
    ok = (gather >= 0)
    gather = np.clip(gather, 0, len(v) - 1)
    ok &= pid[gather] == pid[idx][:, None]
    rel = np.where(ok, logv[gather] - logv[idx][:, None], 0.0)
    yrs = np.where(ok, (date[idx][:, None] - date[gather]).astype(float) / 365.25, 0.0)
    return np.stack([rel, yrs, ok.astype(float)], axis=2).astype("float32")


def _torch():
    import torch
    from torch import nn
    return torch, nn


def make_gru(n_tab, seq_dim=3, hidden=32):
    torch, nn = _torch()

    class GRUNet(nn.Module):
        """GRU over the last 10 valuations + a dense branch for the tabular features."""
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(seq_dim, hidden, batch_first=True)
            self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 64), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(hidden + 64, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, seq, tab):
            _, h = self.gru(seq)
            return self.head(torch.cat([h[-1], self.tab(tab)], dim=1)).squeeze(1)

    return GRUNet()


def train_gru(seq, tab, y, seed, epochs, seq_val=None, tab_val=None, y_val=None,
              patience=5, batch=1024, lr=2e-3, log=print):
    """Train with SmoothL1 loss on the log change. With validation data: early stopping, returns
    (model, best_epoch, val curve). Without: trains exactly `epochs` epochs."""
    torch, nn = _torch()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == "mps":
        torch.mps.manual_seed(seed)
    model = make_gru(tab.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    S, T, Y = (torch.tensor(a, dtype=torch.float32) for a in (seq, tab, y))
    gen = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait, curve = np.inf, 0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(Y), generator=gen).split(batch):
            opt.zero_grad()
            loss = loss_fn(model(S[b].to(dev), T[b].to(dev)), Y[b].to(dev))
            loss.backward()
            opt.step()
        if seq_val is not None:
            mae = float(np.mean(np.abs(predict_gru(model, seq_val, tab_val) - y_val)))
            curve.append(mae)
            if mae < best - 1e-4:
                best, best_ep, wait = mae, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if seq_val is not None:
        model.load_state_dict(best_state)
        return model, best_ep, curve
    return model


def predict_gru(model, seq, tab, batch=8192):
    torch, _ = _torch()
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), batch):
            out.append(model(torch.tensor(seq[i:i + batch], device=dev),
                             torch.tensor(tab[i:i + batch], device=dev, dtype=torch.float32)).cpu().numpy())
    return np.concatenate(out)
