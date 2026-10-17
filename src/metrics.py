"""Evaluation metrics: on the 12-month log change, ranking quality, and the final value in euros."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# "Within ±20%": Transfermarkt rounds values (e.g. €1.0m -> €1.2m), so many errors are EXACTLY 20%.
# We count exactly 20% as within; the tiny tolerance stops float rounding from deciding those ties.
WITHIN = 0.20 + 1e-4

# Columns shown in the paper's main tables (change scale + euros). Value-level R² goes to the appendix.
HEADLINE = ["n", "MAE_change", "R2_change", "Spearman_change", "top10_rise_prec_%", "top10_fall_prec_%",
            "MAE_eur", "MdAPE_%", "within_20%"]


def _top_share(pred_chg, true_chg, frac=0.10):
    """Of the top-`frac` predicted risers, % that really rose; of the top-`frac` predicted fallers,
    % that really fell. Undefined (NaN) when the model predicts the same change for everyone (naive)."""
    if np.ptp(pred_chg) == 0:
        return np.nan, np.nan
    k = int(np.ceil(frac * len(pred_chg)))
    order = np.argsort(pred_chg, kind="stable")
    return 100 * np.mean(true_chg[order[-k:]] > 0), 100 * np.mean(true_chg[order[:k]] < 0)


def evaluate(y_log, pred_log, base_log) -> dict:
    """y_log: true log value at t+12m; pred_log: predicted log value; base_log: log value at t.
    Change metrics compare predicted vs true log change (pred_log - base_log vs y_log - base_log).
    Euro metrics compare exp(pred_log) with the true value."""
    y_log, pred_log, base_log = (np.asarray(a, float) for a in (y_log, pred_log, base_log))
    true_chg, pred_chg = y_log - base_log, pred_log - base_log
    err = pred_chg - true_chg                         # identical to pred_log - y_log
    y, p = np.exp(y_log), np.exp(pred_log)
    ape = np.abs(p / y - 1)
    rise, fall = _top_share(pred_chg, true_chg)
    return {
        "n": len(y),
        # on the 12-month log change (the training target)
        "MAE_change": np.mean(np.abs(err)),
        "RMSE_change": np.sqrt(np.mean(err ** 2)),
        "R2_change": 1 - np.sum(err ** 2) / np.sum((true_chg - true_chg.mean()) ** 2),
        # ranking quality: does the model order players by how their value will change?
        "Spearman_change": np.nan if np.ptp(pred_chg) == 0 else spearmanr(pred_chg, true_chg).statistic,
        "top10_rise_prec_%": rise,
        "top10_fall_prec_%": fall,
        # on the final value in euros
        "MAE_eur": np.mean(np.abs(p - y)),
        "RMSE_eur": np.sqrt(np.mean((p - y) ** 2)),
        "MAPE_%": 100 * np.mean(ape),
        "MdAPE_%": 100 * np.median(ape),
        "within_20%": 100 * np.mean(ape <= WITHIN),
        # value-level R² (appendix only: high mainly because next year's value is close to today's)
        "R2_log": 1 - np.sum(err ** 2) / np.sum((y_log - y_log.mean()) ** 2),
    }


def evaluate_frame(df: pd.DataFrame, pred_log) -> dict:
    """Shortcut for a dataset frame with columns y and log_value."""
    return evaluate(df["y"], pred_log, df["log_value"])


def base_rates(df: pd.DataFrame) -> dict:
    """Share of rows whose value rose / stayed the same / fell over 12 months (reference for precision)."""
    chg = (df["y"] - df["log_value"]).to_numpy()
    return {"rose_%": 100 * np.mean(chg > 0), "unchanged_%": 100 * np.mean(chg == 0),
            "fell_%": 100 * np.mean(chg < 0)}


def table(rows: dict, naive_key=None, appendix=False) -> pd.DataFrame:
    """{name: evaluate(...)} -> rounded DataFrame. With naive_key, adds the skill vs. the naive
    baseline: 1 - MAE_change(model) / MAE_change(naive), in %. The value-level R² is dropped unless
    appendix=True."""
    t = pd.DataFrame(rows).T
    t["n"] = t["n"].astype(int)
    if naive_key is not None:
        t["skill_vs_naive_%"] = 100 * (1 - t["MAE_change"] / t.loc[naive_key, "MAE_change"])
    if not appendix:
        t = t.drop(columns="R2_log")
    return t.round({"MAE_change": 4, "RMSE_change": 4, "R2_change": 4, "R2_log": 4, "Spearman_change": 4,
                    "top10_rise_prec_%": 2, "top10_fall_prec_%": 2, "MAE_eur": 0, "RMSE_eur": 0,
                    "MAPE_%": 2, "MdAPE_%": 2, "within_20%": 2, "skill_vs_naive_%": 1})
