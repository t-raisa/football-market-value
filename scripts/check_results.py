"""Compare every table in results/ with the reference copy in reference_results/.

Numbers must agree to 1e-6 (relative and absolute); text must be identical.
Columns that only record run time ("seconds") are ignored.
Exit code 0 = everything reproduced, 1 = at least one table differs or is missing.
Note: the GRU is trained on the Apple GPU (MPS), which is not guaranteed to be bit-for-bit
deterministic across machines or PyTorch versions, so GRU rows may differ slightly elsewhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {"seconds"}          # wall-clock timings change from run to run
NEW, REF = ROOT / "results", ROOT / "reference_results"


def compare(a: pd.DataFrame, b: pd.DataFrame) -> str:
    a, b = a.drop(columns=[c for c in a.columns if c in IGNORE]), b.drop(columns=[c for c in b.columns if c in IGNORE])
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f"shape/columns differ: {a.shape} vs {b.shape}"
    worst = 0.0
    for c in a.columns:
        x, y = a[c], b[c]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna() | yn.notna()
        if numeric.any():
            if not (xn.isna() == yn.isna()).all():
                return f"column '{c}': missing values differ"
            ok = np.isclose(xn[numeric], yn[numeric], rtol=1e-6, atol=1e-6, equal_nan=True)
            if not ok.all():
                worst = max(worst, float(np.nanmax(np.abs(xn[numeric] - yn[numeric]))))
                return f"column '{c}': {int((~ok).sum())} values differ (max abs diff {worst:.3g})"
        text = ~numeric
        tx = x[text].astype(object).where(x[text].notna(), "<NA>").astype(str).to_numpy()
        ty = y[text].astype(object).where(y[text].notna(), "<NA>").astype(str).to_numpy()
        if not (tx == ty).all():
            return f"column '{c}': text differs"
    return ""


def main() -> int:
    refs = sorted(REF.glob("*.csv")) + sorted(REF.glob("*.json"))
    if not refs:
        print("no reference tables found in reference_results/")
        return 1
    bad = 0
    for ref in refs:
        new = NEW / ref.name
        if not new.exists():
            print(f"MISSING  {ref.name}")
            bad += 1
            continue
        if ref.suffix == ".json":
            msg = "" if new.read_text() == ref.read_text() else "content differs"
        else:
            msg = compare(pd.read_csv(new, dtype=str, keep_default_na=False).replace("", np.nan),
                          pd.read_csv(ref, dtype=str, keep_default_na=False).replace("", np.nan))
        print(f"{'OK      ' if not msg else 'DIFF    '} {ref.name}{'' if not msg else '  -> ' + msg}")
        bad += bool(msg)
    print(f"\n{len(refs) - bad} of {len(refs)} tables reproduced exactly" + ("" if not bad else f"; {bad} differ"))
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
