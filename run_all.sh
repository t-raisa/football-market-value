#!/usr/bin/env bash
# Rebuild every dataset, table and figure of the study from the raw Kaggle zip, then compare
# the result tables with the reference copy in reference_results/.
#
#   ./run_all.sh            run notebooks 01-05 in order, then check
#   ./run_all.sh --check    only compare results/ with reference_results/
#
# Needs: the conda env "football" (environment.yml) registered as a Jupyter kernel named "football",
# and the data zip at data/raw/transfermarkt_davidcariboo.zip (see README).
set -euo pipefail
cd "$(dirname "$0")"

ENV_BIN="${FOOTBALL_ENV_BIN:-$HOME/miniforge3/envs/football/bin}"
JUPYTER="$ENV_BIN/jupyter"
PYTHON="$ENV_BIN/python"
[ -x "$JUPYTER" ] || { echo "jupyter not found at $JUPYTER (set FOOTBALL_ENV_BIN to your env's bin folder)"; exit 1; }

if [ "${1:-}" != "--check" ]; then
  [ -f data/raw/transfermarkt_davidcariboo.zip ] || { echo "missing data/raw/transfermarkt_davidcariboo.zip (see README)"; exit 1; }
  for nb in 01_setup_data 02_features 03_models 04_analysis 05_robustness; do
    start=$(date +%s)
    echo "== running notebooks/$nb.ipynb"
    "$JUPYTER" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=10800 --ExecutePreprocessor.kernel_name=football \
      "notebooks/$nb.ipynb" > /dev/null 2>&1 || { echo "   FAILED: open the notebook to see the error"; exit 1; }
    if grep -q '"output_type": "error"' "notebooks/$nb.ipynb"; then echo "   FAILED: error output in notebook"; exit 1; fi
    echo "   done in $(( $(date +%s) - start )) s"
  done
fi

"$PYTHON" scripts/check_results.py
