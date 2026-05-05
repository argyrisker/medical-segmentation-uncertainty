#!/usr/bin/env bash
set -euo pipefail

python -m src.pipeline \
  --config configs/hippocampus_unet.yaml \
  --run-name hippocampus_unet_baseline \
  --download \
  --split test
