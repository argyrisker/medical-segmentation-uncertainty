#!/usr/bin/env bash
# Stage 2: make the network overfit, then add regularisation back one piece at a time.
# Also trains the baseline on 25% and 50% of the data. Every run has the same
# number of gradient steps per epoch (training.samples_per_epoch).
source "$(dirname "$0")/_common.sh"
have baseline_seed42 || { echo "Run stage1c_baseline.sh first."; exit 1; }

train s2_noreg          --config configs/experiments/s2_noreg.yaml
train s2_aug            --config configs/experiments/s2_aug.yaml
train s2_aug_dropout    --config configs/experiments/s2_aug_dropout.yaml
train s2_aug_dropout_wd --config configs/experiments/s2_aug_dropout_wd.yaml
train baseline_frac0.25 --config configs/experiments/baseline.yaml --data-fraction 0.25
train baseline_frac0.5  --config configs/experiments/baseline.yaml --data-fraction 0.5

RUNS=(s2_noreg s2_aug s2_aug_dropout s2_aug_dropout_wd)
"$PY" -m src.compare_runs --name stage2_overfit --runs "${RUNS[@]}" --reference s2_noreg
"$PY" -m src.plot_curves --runs "${RUNS[@]}" \
  --labels "10 cases, no regularisation" "+ augmentation" "+ dropout" "+ weight decay 1e-2" \
  --compare val_loss train_eval_loss val_dice train_eval_dice --name stage2_overfit

DATA=(s2_aug_dropout baseline_frac0.25 baseline_frac0.5 baseline_seed42)
"$PY" -m src.compare_runs --name stage2_data_size --runs "${DATA[@]}" --reference baseline_seed42
"$PY" -m src.plot_curves --runs "${DATA[@]}" --labels "10 cases" "25% (46 cases)" "50% (91 cases)" "100% (182 cases)" \
  --compare val_dice train_eval_dice --name stage2_data_size
