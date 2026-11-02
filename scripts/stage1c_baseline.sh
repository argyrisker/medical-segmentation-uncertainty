#!/usr/bin/env bash
# Stage 1c: the final baseline with three training seeds. The split stays fixed.
# Gives seed-to-seed spread (is an ablation difference real?) and the ensemble for stage 4.
source "$(dirname "$0")/_common.sh"

for s in 42 43 44; do
  train "baseline_seed$s" --config configs/experiments/baseline.yaml --seed "$s"
done

"$PY" -m src.compare_runs --name stage1c_baseline --runs baseline_seed42 baseline_seed43 baseline_seed44 --group-seeds
"$PY" -m src.plot_curves --runs baseline_seed42 baseline_seed43 baseline_seed44 --compare val_dice --name stage1c_baseline
