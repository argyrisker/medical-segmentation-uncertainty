#!/usr/bin/env bash
# Two-epoch check of every code path on a few cases. Run this first on the GPU
# machine; it takes minutes and catches environment problems before the real runs.
source "$(dirname "$0")/_common.sh"

SMOKE=(--set split.train_cases=6 training.train_eval_cases=3 training.samples_per_epoch=8 training.num_workers=0)
FORCE=1 train smoke_a --config configs/experiments/baseline.yaml --run-name smoke_a --epochs 2 --download "${SMOKE[@]}"
FORCE=1 train smoke_b --config configs/experiments/s3_no_skips.yaml --run-name smoke_b --epochs 2 "${SMOKE[@]}"

"$PY" -m src.evaluate --checkpoint "$CKPT/smoke_a/best_model.pt" --split val --device "$DEVICE" --num-vis 1
"$PY" -m src.infer_uncertainty --checkpoint "$CKPT/smoke_a/best_model.pt" --mode mc_dropout --num-samples 2 --split val --device "$DEVICE" --num-vis 1
"$PY" -m src.infer_uncertainty --checkpoint "$CKPT/smoke_a/best_model.pt" "$CKPT/smoke_b/best_model.pt" \
  --mode ensemble --run-name smoke_ensemble --split val --device "$DEVICE" --num-vis 1
"$PY" -m src.calibration --checkpoint "$CKPT/smoke_a/best_model.pt" --split val --temperature-scaling --device "$DEVICE"
"$PY" -m src.compare_runs --name smoke --runs smoke_a smoke_b --reference smoke_a
"$PY" -m src.plot_curves --runs smoke_a smoke_b --compare val_dice --name smoke
"$PY" -m src.compare_uncertainty --split val --name smoke_uncertainty --entries smoke_a:mc_dropout smoke_ensemble:ensemble
echo "Smoke test passed. Check timing in outputs/checkpoints/smoke_a/training_history.csv (epoch_time_s)."
