#!/usr/bin/env bash
# Rebuild every dataset, table and figure of the study from the raw Kaggle zip, then compare
# the result tables with the reference copy in reference_results/.
#
#   ./run_all.sh            check the data version, run notebooks 01-06 in order, then check all tables
#   ./run_all.sh --check    only compare results/ with reference_results/
#
# Needs: the conda env "football" (environment.yml) registered as a Jupyter kernel named "football",
# and the data zip at data/raw/transfermarkt_davidcariboo.zip (see README).
set -uo pipefail
cd "$(dirname "$0")"

ENV_BIN="${FOOTBALL_ENV_BIN:-$HOME/miniforge3/envs/football/bin}"
JUPYTER="$ENV_BIN/jupyter"
PYTHON="$ENV_BIN/python"
[ -x "$JUPYTER" ] || { echo "jupyter not found at $JUPYTER (set FOOTBALL_ENV_BIN to your env's bin folder)"; exit 1; }

if [ "${1:-}" != "--check" ]; then
  [ -f data/raw/transfermarkt_davidcariboo.zip ] || { echo "missing data/raw/transfermarkt_davidcariboo.zip (see README)"; exit 1; }
  echo "== checking the dataset version (SHA-256)"
  if ! "$PYTHON" scripts/verify_data.py > /dev/null; then
    "$PYTHON" scripts/verify_data.py | grep -v "^OK" || true
    if [ "${ALLOW_OTHER_DATA:-0}" != "1" ]; then
      echo "   stopping: this is not the dataset version used in the paper (set ALLOW_OTHER_DATA=1 to run anyway)"; exit 1
    fi
    echo "   continuing with a different dataset version (ALLOW_OTHER_DATA=1): results will differ from the paper"
  else
    echo "   same version as the paper"
  fi
  for nb in 01_setup_data 02_features 03_models 04_analysis 05_robustness 06_review_checks; do
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
status=$?
"$PYTHON" scripts/make_paper_assets.py      # refresh the paper's numbers and tables
if [ -x "$ENV_BIN/tectonic" ]; then          # optional: compile the paper if tectonic is installed
  (cd paper && "$ENV_BIN/tectonic" main.tex > /dev/null 2>&1 && "$ENV_BIN/tectonic" supplement.tex > /dev/null 2>&1) \
    && echo "compiled paper/main.pdf and paper/supplement.pdf" || echo "paper compile FAILED (run tectonic in paper/ to see why)"
fi
exit $status
