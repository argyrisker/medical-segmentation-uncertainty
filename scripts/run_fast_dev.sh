#!/usr/bin/env bash
set -euo pipefail


python -m src.pipeline \
  --config configs/hippocampus_unet.yaml \
  --run-name fast_dev \
  --download \
  --fast-dev-run \
  --split val
