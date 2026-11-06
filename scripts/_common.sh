# Shared helpers for the stage scripts. Source this file; do not run it.
#
# Environment variables:
#   PYTHON  python executable (default: python)
#   DEVICE  auto | cuda | cuda:1 | cpu (default: auto)
#   FORCE=1 retrain runs that already finished
#   EPOCHS  train every run for this many epochs instead of the config value

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PYTHON:-python}"
DEVICE="${DEVICE:-auto}"
CKPT=outputs/checkpoints
mkdir -p outputs/logs

# train <expected_run_name> <src.train arguments...>
# Skips the run if it already finished (summary.json exists).
train() {
  local run="$1"; shift
  if [[ -f "${CKPT}/${run}/summary.json" && "${FORCE:-0}" != "1" ]]; then
    echo "== ${run}: already trained, skipping (FORCE=1 to retrain)"
    return
  fi
  echo "== ${run}: training (log: outputs/logs/${run}.log)"
  # An explicit --epochs inside "$@" (smoke test) comes later and wins.
  "$PY" -m src.train --device "$DEVICE" ${EPOCHS:+--epochs "$EPOCHS"} "$@" 2>&1 | tee "outputs/logs/${run}.log"
  [[ -f "${CKPT}/${run}/summary.json" ]] || { echo "Expected run ${run} was not produced"; exit 1; }
}

# Name of the stage 1a config that configs/experiments/baseline.yaml inherits from.
baseline_parent() {
  sed -n 's/^base:[[:space:]]*\(.*\)\.yaml.*/\1/p' configs/experiments/baseline.yaml
}

have() { [[ -f "${CKPT}/$1/best_model.pt" ]]; }
