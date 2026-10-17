"""Analysis helpers: player-level (cluster) bootstrap for confidence intervals."""
import numpy as np
import pandas as pd

from .config import SEED

N_BOOT = 1000


def cluster_bootstrap_means(player_id, values: pd.DataFrame, B=N_BOOT, seed=SEED) -> np.ndarray:
    """Resample PLAYERS (not rows) with replacement, B times. A drawn player brings all of their rows.
    For every draw and every column, returns the mean over the drawn rows.
    Columns are resampled together (same players), so differences between columns are paired.
    Returns an array of shape (B, n_columns)."""
    players, inv = np.unique(np.asarray(player_id), return_inverse=True)
    n = len(players)
    sums = np.zeros((n, values.shape[1]))
    np.add.at(sums, inv, values.to_numpy(dtype=float))
    counts = np.bincount(inv, minlength=n).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float64)
    return (w @ sums) / (w @ counts)[:, None]


def ci(draws, level=0.95):
    """Percentile confidence interval."""
    a = (1 - level) / 2
    return np.quantile(draws, a), np.quantile(draws, 1 - a)


def fmt_ci(point, lo, hi, digits=3):
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def verdict(lo, hi, name_a, name_b):
    """Rule fixed in advance: if the 95% CI of (MAE_a - MAE_b) contains 0 -> no clear difference."""
    if lo <= 0 <= hi:
        return "no clear difference"
    return f"{name_a} better" if hi < 0 else f"{name_b} better"
