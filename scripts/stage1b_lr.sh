#!/usr/bin/env bash
# Stage 1b: learning-rate sweep on the chosen patch strategy (lr 3e-4 is the stage 1a run).
source "$(dirname "$0")/_common.sh"

PARENT="$(baseline_parent)"
have "$PARENT" || { echo "Run stage1a_patch.sh first ($PARENT missing)."; exit 1; }

train s1b_lr1e-4 --config configs/experiments/s1b_lr1e-4.yaml
train s1b_lr1e-3 --config configs/experiments/s1b_lr1e-3.yaml

"$PY" -m src.compare_runs --name stage1b_lr --runs s1b_lr1e-4 "$PARENT" s1b_lr1e-3 --reference "$PARENT"
"$PY" -m src.plot_curves --runs s1b_lr1e-4 "$PARENT" s1b_lr1e-3 --labels "lr 1e-4" "lr 3e-4" "lr 1e-3" \
  --compare val_dice val_loss train_eval_loss --name stage1b_lr

echo
echo "Next: set training.lr in configs/experiments/baseline.yaml to the best value"
echo "(see outputs/tables/stage1b_lr.md), then run stage1c_baseline.sh."
