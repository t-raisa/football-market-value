"""Project-wide settings: paths, random seed, split dates, plotting defaults."""
from pathlib import Path
import os
import random

import numpy as np

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
RAW_ZIP = RAW / "transfermarkt_davidcariboo.zip"

for _d in (RAW, PROCESSED, FIGURES, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- reproducibility ----------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use (Python, NumPy, PyTorch)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


# ---------- forecasting setup (used from notebook 02 on) ----------
HORIZON_DAYS = 365        # predict value ~12 months ahead
HORIZON_TOL_DAYS = 60     # accept a future valuation within +/- 60 days
TRAIN_END = "2023-07-01"  # train: t < TRAIN_END
VAL_END = "2024-07-01"    # validation: TRAIN_END <= t < VAL_END; test: t >= VAL_END

# ---------- analysis groups (fixed before looking at any results) ----------
VALUE_BINS = [0, 1e6, 10e6, float("inf")]           # value at t, EUR
VALUE_LABELS = ["cheap (<€1m)", "medium (€1–10m)", "expensive (≥€10m)"]
AGE_BINS = [0, 21, 26, 30, 100]                     # age at t, years: <21, 21-25, 26-29, 30+
AGE_LABELS = ["<21", "21–25", "26–29", "30+"]

# ---------- figures ----------
FIG_DPI = 300
# Fixed categorical order (colorblind-validated); always assign in this order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#52514e"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply_plot_style() -> None:
    """Clean, paper-ready matplotlib defaults: thin marks, light grid, no top/right spines."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": FIG_DPI, "font.size": 9,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.edgecolor": TEXT_SECONDARY, "axes.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 2, "legend.frameon": False,
    })


def savefig(fig, name: str) -> Path:
    """Save a matplotlib figure as a 300-dpi PNG in figures/."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    return path
