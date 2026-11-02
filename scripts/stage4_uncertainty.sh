#!/usr/bin/env bash
# Stage 4: uncertainty and calibration. SPLIT=val (default) while exploring;
# stage5_test.sh calls this again with SPLIT=test for the final numbers.
source "$(dirname "$0")/_common.sh"
SPLIT="${SPLIT:-val}"
B=baseline_seed42
for s in 42 43 44; do have "baseline_seed$s" || { echo "Run stage1c_baseline.sh first."; exit 1; }; done

"$PY" -m src.infer_uncertainty --checkpoint "$CKPT/$B/best_model.pt" --mode deterministic --split "$SPLIT" --device "$DEVICE"
"$PY" -m src.infer_uncertainty --checkpoint "$CKPT/$B/best_model.pt" --mode mc_dropout --split "$SPLIT" --device "$DEVICE"
"$PY" -m src.infer_uncertainty --mode ensemble --run-name baseline_ensemble --split "$SPLIT" --device "$DEVICE" \
  --checkpoint "$CKPT/baseline_seed42/best_model.pt" "$CKPT/baseline_seed43/best_model.pt" "$CKPT/baseline_seed44/best_model.pt"
ENTRIES=("$B:deterministic" "$B:mc_dropout" "baseline_ensemble:ensemble")

if have s3_dropout0.3; then
  "$PY" -m src.infer_uncertainty --checkpoint "$CKPT/s3_dropout0.3/best_model.pt" --mode mc_dropout --split "$SPLIT" --device "$DEVICE"
  ENTRIES+=("s3_dropout0.3:mc_dropout")
fi

# Temperature is always fitted on validation voxels. On SPLIT=val the "after" numbers are in-sample.
"$PY" -m src.calibration --checkpoint "$CKPT/$B/best_model.pt" --split "$SPLIT" --temperature-scaling --device "$DEVICE"
"$PY" -m src.analysis --run-name "$B" --split "$SPLIT" --mode mc_dropout
"$PY" -m src.compare_uncertainty --split "$SPLIT" --name "uncertainty_$SPLIT" --entries "${ENTRIES[@]}"
