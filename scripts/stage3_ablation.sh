#!/usr/bin/env bash
# Stage 3: ablations. Each run changes one thing against baseline_seed42.
# PRIORITY=1 runs only the five most important ones (default: all, in priority order).
source "$(dirname "$0")/_common.sh"
have baseline_seed42 || { echo "Run stage1c_baseline.sh first."; exit 1; }

P1=(s3_no_skips s3_aug_none s3_aug_anatomical s3_loss_ce s3_loss_dice)
P2=(s3_dropout0 s3_dropout0.3 s3_no_residual s3_depth3)
P3=(s3_batchnorm)
case "${PRIORITY:-3}" in
  1) RUNS=("${P1[@]}") ;;
  2) RUNS=("${P1[@]}" "${P2[@]}") ;;
  *) RUNS=("${P1[@]}" "${P2[@]}" "${P3[@]}") ;;
esac

for r in "${RUNS[@]}"; do
  train "$r" --config "configs/experiments/$r.yaml"
done

DONE=(baseline_seed42)
for r in "${P1[@]}" "${P2[@]}" "${P3[@]}"; do have "$r" && DONE+=("$r"); done
"$PY" -m src.compare_runs --name stage3_ablation --runs "${DONE[@]}" --reference baseline_seed42
"$PY" -m src.plot_curves --runs "${DONE[@]}" --compare val_dice val_loss --name stage3_ablation
echo "Seed spread of the baseline, to judge whether a difference is real: outputs/tables/stage1c_baseline.md"
