# Medical Segmentation Uncertainty

This repository trains a 3D U-Net baseline on the Medical Segmentation Decathlon Task04 Hippocampus dataset, evaluates segmentation quality, and estimates predictive uncertainty with Monte Carlo dropout.

## Goal

Research question:

> Can uncertainty estimation improve the reliability of 3D medical image segmentation under limited-data conditions?

## Pipeline

The pipeline connects:

1. MONAI preprocessing and augmentation
2. 3D U-Net training
3. Basic validation Dice during training
4. Sliding-window inference
5. Dice, HD95, sensitivity, specificity
6. Prediction visualizations
7. MC dropout uncertainty
8. Entropy and variance uncertainty maps
9. Calibration analysis
10. Dice-vs-uncertainty plots
11. Optional low-data experiments

## Environment

Create the environment first:

```bash
bash setup_env.sh gpu
conda activate medseg-uncertainty
```

If you do not have a usable NVIDIA GPU:

```bash
bash setup_env.sh cpu
```

Do not train serious 3D models on CPU unless this is only a smoke test.

## Fast smoke test

Run this before full training:

```bash
bash scripts/run_fast_dev.sh
```

This uses only 2 epochs. It is not a result. It only checks that the code path works.

## Baseline full pipeline

```bash
bash scripts/run_baseline.sh
```

This runs:

```text
train -> evaluate -> uncertainty inference -> calibration -> analysis plots
```

Outputs:

```text
outputs/checkpoints/<run_name>/best_model.pt
outputs/metrics/<run_name>/
outputs/predictions/<run_name>/
outputs/uncertainty_maps/<run_name>/
reports/figures/<run_name>/
```

## Strong version: low-data experiments

```bash
bash scripts/run_strong_low_data.sh
```

This trains and evaluates:

```text
100% training data
50% training data
25% training data
10% training data
```

The validation and internal test sets stay fixed. That matters; otherwise the comparison is invalid.

## Individual commands

Train:

```bash
python -m src.train \
  --config configs/hippocampus_unet.yaml \
  --run-name hippocampus_unet_baseline \
  --data-fraction 1.0 \
  --download
```

Evaluate:

```bash
python -m src.evaluate \
  --config configs/hippocampus_unet.yaml \
  --checkpoint outputs/checkpoints/hippocampus_unet_baseline/best_model.pt \
  --split test
```

Uncertainty:

```bash
python -m src.infer_uncertainty \
  --config configs/hippocampus_unet.yaml \
  --checkpoint outputs/checkpoints/hippocampus_unet_baseline/best_model.pt \
  --split test
```

Calibration:

```bash
python -m src.calibration \
  --config configs/hippocampus_unet.yaml \
  --checkpoint outputs/checkpoints/hippocampus_unet_baseline/best_model.pt \
  --split test
```

Analysis:

```bash
python -m src.analysis \
  --config configs/hippocampus_unet.yaml \
  --run-name hippocampus_unet_baseline \
  --split test
```

## Important methodological choices

### Internal labelled test split

The official MSD test set is unlabelled. This project creates a fixed internal split from the labelled training data:

```yaml
val_frac: 0.15
test_frac: 0.15
```

This is not perfect, but it is honest and reproducible.

### MRI normalization

This project uses nonzero z-score normalization:

```python
NormalizeIntensityd(nonzero=True, channel_wise=True)
```

Do not use CT-style fixed Hounsfield windows for MRI.

### Foreground-biased crop sampling

Hippocampus occupies a tiny fraction of each volume. Random crops would mostly be background, so the training transform uses:

```python
RandCropByPosNegLabeld(pos=2.0, neg=1.0)
```

## Metrics

Reported metrics:

- Dice
- HD95
- Sensitivity
- Specificity
- Expected Calibration Error
- Entropy/variance uncertainty summaries

Specificity is included, but do not headline it. It is often inflated by background voxels in segmentation.

## Limitations

- Single MSD task
- Internal split, not external validation
- Single model family
- MC dropout is only an approximate uncertainty method
- Uncertainty maps are not clinical safety guarantees

The purpose of this repo is to demonstrate a clean medical imaging ML workflow, not to claim clinical deployment readiness.
