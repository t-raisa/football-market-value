"""Simple, transparent baselines for the 12-month log change. Each predicts the change y - log_value.

Fitted on training rows only (like every other model): group statistics come from the training targets;
value bins are training-set quantiles. All inputs (age, value, league, last change) are known at t.
"""
import numpy as np
import pandas as pd

AGE_EDGES = [0, 21, 24, 27, 30, 33, 100]                   # <21, 21-23, 24-26, 27-29, 30-32, 33+
AGE_NAMES = ["<21", "21-23", "24-26", "27-29", "30-32", "33+"]
N_VALUE_BINS = 5                                           # value quintiles of the training data
MIN_CELL = 30                                              # smaller cells fall back to a coarser group


def _change(d):
    return (d["y"] - d["log_value"]).astype("float64")


class AgeValueCurve:
    """Average 12-month change by age group x value level (optionally x league at t).
    Fallback chain for small cells (< MIN_CELL training rows): age x value x league -> age x value -> age -> all."""

    def __init__(self, by_league=False, stat="median"):
        """stat: "mean", "median", or a number in (0, 1) for that quantile (used for forecast intervals)."""
        self.by_league, self.stat = by_league, stat

    def _agg(self):
        return (lambda c: c.quantile(self.stat)) if isinstance(self.stat, float) else self.stat

    def _keys(self, d):
        age = pd.cut(d["age"].astype(float), AGE_EDGES, labels=False, right=False)
        val = np.searchsorted(self.value_edges, d["log_value"].astype(float).to_numpy(), side="right")
        return pd.DataFrame({"age": age.to_numpy(), "val": val, "lg": d["league_t"].astype(str).to_numpy()},
                            index=d.index)

    def fit(self, d):
        self.value_edges = np.quantile(d["log_value"].astype(float), np.linspace(0, 1, N_VALUE_BINS + 1)[1:-1])
        k = self._keys(d).assign(chg=_change(d))
        agg = lambda cols: k.groupby(cols, dropna=True)["chg"].agg(val=self._agg(), size="size")
        self.tables = [t[t["size"] >= MIN_CELL]["val"] for t in
                       ([agg(["age", "val", "lg"])] if self.by_league else []) + [agg(["age", "val"]), agg(["age"])]]
        self.levels = ([["age", "val", "lg"]] if self.by_league else []) + [["age", "val"], ["age"]]
        self.overall = float(k["chg"].quantile(self.stat) if isinstance(self.stat, float) else k["chg"].agg(self.stat))
        return self

    def predict(self, d):
        k = self._keys(d)
        out = pd.Series(np.nan, index=d.index)
        for cols, table in zip(self.levels, self.tables):
            idx = pd.MultiIndex.from_frame(k[cols]) if len(cols) > 1 else pd.Index(k[cols[0]])
            vals = pd.Series(table.reindex(idx).to_numpy(), index=d.index)
            out = out.fillna(vals)
        return out.fillna(self.overall).to_numpy()


class Momentum:
    """The last observed change continues: change since the previous valuation, scaled to 12 months
    (365 / gap, with the gap floored at 90 days so very close valuations do not explode), times `damp`.
    Players without an earlier valuation get 'no change'."""

    def __init__(self, damp=1.0, min_gap_days=90):
        self.damp, self.min_gap = damp, min_gap_days

    def fit(self, d):
        return self

    def predict(self, d):
        chg = d["change_since_prev_valuation"].astype(float)
        gap = np.maximum(d["days_since_prev_valuation"].astype(float), self.min_gap)
        return (self.damp * chg * 365.0 / gap).fillna(0.0).to_numpy()


class NoChange:
    def fit(self, d):
        return self

    def predict(self, d):
        return np.zeros(len(d))


def simple_baselines(age_stat="median"):
    """Name -> unfitted baseline. Names are ASCII so they can be used as column names."""
    return {
        "Naive (no change)": NoChange(),
        "Age-value curve": AgeValueCurve(by_league=False, stat=age_stat),
        "Age-value-league curve": AgeValueCurve(by_league=True, stat=age_stat),
        "Momentum": Momentum(damp=1.0),
        "Momentum (damped)": Momentum(damp=0.5),
    }


SIMPLE_NAMES = list(simple_baselines())
