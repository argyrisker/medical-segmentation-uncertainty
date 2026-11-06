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

## Course project experiments (v2)

The staged experiment plan for the deep-learning course lives in
`docs/course_project/plan.md`. Each stage is one script; each run is one small
YAML file in `configs/experiments/` that inherits from a base with `base:` and
lists only what it changes. Finished runs are skipped when a script is rerun.

To run everything unattended on a GPU server, see
`docs/course_project/running_on_gpu.md`. In short:

```bash
nohup bash scripts/run_all.sh > run_all.log 2>&1 &   # or: sbatch scripts/run_all.sh
```

Or stage by stage:

```bash
bash scripts/smoke_test.sh        # minutes; run first on the GPU machine
bash scripts/stage1a_patch.sh     # v1 rerun + two patch strategies
#   then edit configs/experiments/baseline.yaml (base: winner)
bash scripts/stage1b_lr.sh        # learning-rate sweep
#   then set training.lr in configs/experiments/baseline.yaml
bash scripts/stage1c_baseline.sh  # final baseline, 3 training seeds
bash scripts/stage2_overfit.sh    # overfitting study + data-size runs
bash scripts/stage3_ablation.sh   # ablations (PRIORITY=1 for the top five)
bash scripts/stage4_uncertainty.sh  # uncertainty + calibration on val
bash scripts/stage5_test.sh       # the only use of the test set
```

Set `PYTHON=...` to choose the interpreter and `DEVICE=cuda:1` to pick a GPU.
Every training run writes to `outputs/checkpoints/<run>/`:

| File | Content |
|---|---|
| `training_history.csv` | Per epoch: lr, train loss, train and val loss/Dice on whole volumes |
| `best_model.pt` | Weights at the best validation Dice, with the exact config used |
| `summary.json` | Best epoch, best val Dice, parameter count, run time |
| `val_per_case_best.csv` | Per-case validation Dice at the best epoch, for paired tests |
| `config_used.yaml` | The fully resolved config |

Tables go to `outputs/tables/`, figures to `reports/figures/`, and logs to `outputs/logs/`.

What changed from v1:

- **Own 3D U-Net** (`model.arch: custom`) with switches for skip connections, residual blocks, depth, dropout and normalisation.
- **Train and inference windows match.** v1 trained on 32³ patches but predicted with 64³ windows that were mostly zero padding.
- **Full logging**: validation loss and train Dice on whole volumes every epoch, not only training loss.
- **Cosine learning-rate schedule** and more epochs; v1 was still improving when training stopped.
- **Split seed separate from training seed**, so models trained with different seeds share one test set.
- **New metrics**: precision, voxel confusion matrix, error-detection ROC/AUROC, temperature scaling.
- **No 90-degree rotations in v2.** They swap the 48 and 64 sides of the whole-scan patch, so batches cannot be stacked. Flips on all three axes remain.
- **No foreground crop.** The scans are pre-cropped; in v1 the crop only added a 5-voxel zero border (MONAI 1.5 pads when the margin passes the image edge).

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

Every command takes `--help`. Evaluation scripts rebuild the model from the
config stored inside the checkpoint, so they work for any experiment.

Train (any config; `--set` overrides single values, `--seed` changes the training seed only):

```bash
python -m src.train --config configs/experiments/baseline.yaml --seed 42 --download
python -m src.train --config configs/experiments/baseline.yaml --set training.lr=0.001 --run-name lr_test
```

Evaluate on train, val or test (metrics, confusion matrix, example figures):

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/baseline_seed42/best_model.pt --split val
```

Uncertainty (`deterministic`, `mc_dropout`, or `ensemble` with several checkpoints):

```bash
python -m src.infer_uncertainty --checkpoint outputs/checkpoints/baseline_seed42/best_model.pt --mode mc_dropout --split val
```

Calibration, with temperature scaling fitted on the validation set:

```bash
python -m src.calibration --checkpoint outputs/checkpoints/baseline_seed42/best_model.pt --split test --temperature-scaling
```

Learning curves and comparison tables:

```bash
python -m src.plot_curves --runs baseline_seed42 s3_no_skips --compare val_dice val_loss --name skips
python -m src.compare_runs --runs baseline_seed42 s3_no_skips --reference baseline_seed42 --name skips
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

## License

The source code in this repository is licensed under the MIT License.

This project uses the Medical Segmentation Decathlon Task04 Hippocampus dataset. Dataset-derived figures, including MRI overlays and segmentation visualizations, are included only for research and portfolio documentation purposes and should be attributed to the original dataset source according to its license terms.

Model checkpoints, raw dataset files, prediction arrays, and generated experiment outputs are not included in this repository.
