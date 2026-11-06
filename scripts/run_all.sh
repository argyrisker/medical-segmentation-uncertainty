#!/usr/bin/env bash
# Runs the whole experiment plan unattended, in order. Start it from the
# repository folder:
#
#   nohup bash scripts/run_all.sh > run_all.log 2>&1 &    # plain GPU machine
#   sbatch scripts/run_all.sh                              # Slurm cluster
#
# It uses the default baseline, so the edits suggested between the stage 1
# scripts are optional. Finished runs are skipped, so after a crash or a
# disconnect just start it again.
#
# Options (environment variables):
#   EPOCHS=100     epochs per training run (default: the config, 200)
#   PRIORITY=2     ablations: 1 = top five (default), 2 = nine, 3 = all ten
#   DEVICE=cuda:1  which GPU
#   PYTHON=...     which python
#
#SBATCH --job-name=hippocampus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=run_all_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PRIORITY="${PRIORITY:-1}"

stage() {
  echo
  echo "######## $1 started $(date '+%a %d %b %H:%M')"
  bash "scripts/$1"
}

stage smoke_test.sh
stage stage1a_patch.sh
stage stage1b_lr.sh
stage stage1c_baseline.sh
stage stage2_overfit.sh
stage stage3_ablation.sh
stage stage4_uncertainty.sh
stage stage5_test.sh

echo
echo "######## All stages finished $(date '+%a %d %b %H:%M')"
echo "Tables: outputs/tables/   Plots: reports/figures/   Logs: outputs/logs/"
