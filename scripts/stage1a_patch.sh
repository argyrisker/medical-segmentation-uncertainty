#!/usr/bin/env bash
# Stage 1a: reproduce v1 with full logging, then compare the two patch strategies.
source "$(dirname "$0")/_common.sh"

train v1_legacy     --config configs/experiments/v1_legacy.yaml --download
train s1a_patch32   --config configs/experiments/s1a_patch32.yaml
train s1a_patchfull --config configs/experiments/s1a_patchfull.yaml

"$PY" -m src.compare_runs --name stage1a_patch --runs v1_legacy s1a_patch32 s1a_patchfull --reference v1_legacy
"$PY" -m src.plot_curves --runs v1_legacy s1a_patch32 s1a_patchfull --compare val_dice val_loss --name stage1a_patch

echo
echo "Next: in configs/experiments/baseline.yaml, set 'base:' to the patch config"
echo "with the higher best_val_dice (see outputs/tables/stage1a_patch.md), then run stage1b_lr.sh."
