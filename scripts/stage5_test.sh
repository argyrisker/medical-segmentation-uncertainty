#!/usr/bin/env bash
# Stage 5: the only time the test set is used. Evaluates every finished model,
# reports train/val/test for the baseline, and reruns stage 4 on test.
source "$(dirname "$0")/_common.sh"
have baseline_seed42 || { echo "Run stage1c_baseline.sh first."; exit 1; }

RUNS=(v1_legacy baseline_seed42 baseline_seed43 baseline_seed44
      s3_no_skips s3_aug_none s3_aug_anatomical s3_loss_ce s3_loss_dice
      s3_dropout0 s3_dropout0.3 s3_no_residual s3_depth3 s3_batchnorm)
DONE=()
for r in "${RUNS[@]}"; do
  if have "$r"; then
    "$PY" -m src.evaluate --checkpoint "$CKPT/$r/best_model.pt" --split test --device "$DEVICE"
    DONE+=("$r")
  fi
done

# Train / validation / test performance of the final model (course checklist item).
for split in train val; do
  "$PY" -m src.evaluate --checkpoint "$CKPT/baseline_seed42/best_model.pt" --split "$split" --device "$DEVICE" --num-vis 0
done

"$PY" -m src.compare_runs --name final_test --runs "${DONE[@]}" --reference baseline_seed42 --group-seeds
SPLIT=test bash scripts/stage4_uncertainty.sh
"$PY" -m src.plot_curves --runs baseline_seed42 v1_legacy

echo
echo "Slide material:"
echo "  tables:   outputs/tables/*.md"
echo "  curves:   reports/figures/curves/"
echo "  test:     outputs/metrics/baseline_seed42/test_confusion_matrix.png, outputs/predictions/baseline_seed42/"
echo "  uncert.:  reports/figures/uncertainty/, reports/figures/baseline_seed42/"
