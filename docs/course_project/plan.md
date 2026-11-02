# Course project plan: 3D hippocampus segmentation

Course: Introduction to Deep Learning 2026, final project option 3, "bring your own data/task".

| Date | What |
|---|---|
| Done | Proposal sent and approved |
| Tue Nov 10 | Send slides to mahmoud.Rahat@hh.se. This also books the presentation slot, first come first served |
| Fri Nov 13, 13:00–16:00 | 15-minute slot: 12 min talk, 3 min questions. Attend the whole session |

## 1. Does this project fit?

Yes, with a change of emphasis.

- **It is a deep neural network.** A 3D U-Net is a CNN with skip connections, which is exactly what the course hints push for in its own segmentation project.
- **It is healthcare data** and it is option 3, so a proposal email is required before you start.
- **The emphasis must shift.** The current report is about uncertainty. The course grades deep-learning methodology: learning curves, overfitting analysis, ablation, hyperparameter tuning. Uncertainty becomes the extra angle, about 2–3 slides.
- **Be open about prior work.** Say in the proposal that this builds on your earlier pipeline and list what is new for the course.

## 2. Gap analysis against the course checklist

| Course asks for | Current repo | Gap |
|---|---|---|
| Describe the network | 3D U-Net, MONAI | None, but add a diagram and parameter count |
| Optimization and regularization | AdamW, weight decay, dropout, augmentation, instance norm | No learning-rate schedule |
| Train / val / test performance | Test metrics. Train loss only. Val Dice only | No train Dice, no val loss |
| Test metrics: accuracy, precision, recall, F1, confusion matrix, ROC | Dice, HD95, sensitivity, specificity | Add precision, confusion matrix, ROC. Dice equals F1 |
| Training/validation loss plot | Not produced | Validation loss is never computed |
| Train for more than 50 epochs | 100 epochs | Best val Dice at epoch 96 of 100, so the model is undertrained |
| Make it overfit and analyse | Missing | Whole experiment needed |
| Ablation study | Missing. Low-data script exists, no results | Whole experiment needed |
| Hyperparameter tuning | Missing | Whole experiment needed |
| Literature | Missing | See section 7 |

## 3. Problems in the current pipeline

The current test Dice is 0.716. nnU-Net reaches about 0.89–0.90 on this task. Those numbers come from the official test set, so they are not directly comparable, but the gap is far too large to be explained by the split. These are the likely causes, in order of expected impact. Fixing them is the first "improvement" story of the talk.

1. **Undertraining.** Learning rate is a constant 1e-4 with no schedule, and validation Dice was still rising at epoch 96. See `configs/hippocampus_unet.yaml`.
2. **Train/inference window mismatch.** Training uses 32³ patches, but inference uses 64³ windows. The volumes are only about 35×50×36, so MONAI's sliding-window inference zero-pads every window to 64³. Roughly three quarters of each inference window is padding the network never saw in training, and instance normalisation statistics shift accordingly. Fix: make `roi_size` equal `patch_size`, and compare 32³ against a patch that covers the whole volume, such as 48×64×48. nnU-Net plans a patch of about 40×56×40 for this task.
3. **Foreground crop after z-scoring.** Checked on all 260 scans: this does no harm to the labels. Bright voxels touch every image edge, so the crop box is the whole scan, and since MONAI 1.5 the 5-voxel margin then *pads* instead of cropping. Every v1 volume gained a 5-voxel zero border per side and no labelled voxel was lost. v2 drops the crop, because the scans are already cropped to at most 43×59×47.
4. **Anterior–posterior flips.** Labels 1 and 2 are anterior and posterior hippocampus. After RAS reorientation, axis 1 is the anterior–posterior axis, and the pipeline flips it and rotates in the axial plane. This produces anatomically impossible training images. nnU-Net mirrors all axes and still does well, so treat this as an ablation, not a bug.
5. **No validation loss and no train Dice.** Without these you cannot draw the required loss plot or diagnose overfitting.

## 4. Experiment plan

**Status: stage 0 is done.** Every stage below is one script in `scripts/`, and every run is one YAML file in `configs/experiments/`. Run `scripts/smoke_test.sh` first on the GPU machine, and read the epoch time it prints to budget the rest.

Early sanity check, run on CPU while testing the code. These are validation numbers from a short run, not results for the slides:

| Run | Epochs | Best validation Dice |
|---|---|---|
| v1 (original report) | 100 | 0.700 |
| v2 baseline | 8 | 0.787 |

The v1 rerun also reproduced the original first-epoch training loss exactly (2.0226), so the `v1_legacy` run is a faithful reference.

Ground rules:

- Make every tuning and ablation decision on the validation set. Touch the test set once, for the final models.
- Keep the split and seed fixed across runs. Use 3 seeds for the final baseline. The test set has only 39 cases, so report mean ± SD and a paired Wilcoxon test for key comparisons.
- Log per epoch: train loss, val loss, train Dice on a fixed subset of about 20 training cases, val Dice, learning rate.
- Set `cache_rate: 1.0`. The whole dataset fits in memory easily.
- Time the first full run, then budget the rest. Queue runs overnight.

### Stage 0: code changes (done)

| File | Change |
|---|---|
| `src/train.py` | Validation loss and train Dice/loss on whole volumes every epoch, LR logging, cosine schedule, `--seed` and `--set` overrides, history written every epoch, `summary.json` |
| `src/models.py` | Own `UNet3D` with switches for skip connections, residual blocks, depth, dropout and normalisation |
| `src/transforms.py` | Crop mode and augmentation level (`full`, `anatomical`, `none`) as config options |
| `src/losses.py` | Dice+CE, Dice only or CE only |
| `src/metrics.py`, `src/evaluate.py` | Precision, voxel confusion matrix, train/val/test evaluation |
| `src/infer_uncertainty.py` | Deterministic, MC dropout and ensemble modes, error-detection AUROC and ROC curve |
| `src/calibration.py` | Temperature scaling fitted on validation |
| `src/plot_curves.py`, `src/compare_runs.py`, `src/compare_uncertainty.py` | Learning curves, comparison tables with paired Wilcoxon tests, ROC overlays |
| `src/config_utils.py`, `src/datasets.py` | Config inheritance, split seed kept separate from the training seed |

### Stage 1: fixed baseline and tuning

| Script | Runs | Purpose |
|---|---|---|
| `stage1a_patch.sh` | `v1_legacy`, `s1a_patch32`, `s1a_patchfull` | Reference curves for v1, then fixes 1–2 with a 32³ patch or a whole-scan 48×64×48 patch |
| `stage1b_lr.sh` | `s1b_lr1e-4`, `s1b_lr1e-3` | Learning-rate sweep. The stage 1a winner is the lr 3e-4 point |
| `stage1c_baseline.sh` | `baseline_seed42/43/44` | Final baseline with three training seeds on the same split |

Between scripts you make two one-line edits to `configs/experiments/baseline.yaml`, which the scripts print. Every later stage inherits from that file. The seed spread from stage 1c tells you how large an ablation difference must be to mean anything.

### Stage 2: overfitting study

`stage2_overfit.sh` trains, all on 10 cases with the same number of gradient steps per epoch as the full data:

1. `s2_noreg`: no augmentation, dropout 0, weight decay 0. Expect train Dice near 1 and validation loss rising.
2. `s2_aug`: plus augmentation.
3. `s2_aug_dropout`: plus dropout 0.1. This is the baseline's regularisation; its weight decay of 1e-5 is too small to matter.
4. `s2_aug_dropout_wd`: plus a real weight decay of 1e-2.

It also trains the baseline on 25% and 50% of the data. With the 10-case and full runs, that gives a data-size curve.

If the full-data model will not overfit, say why: random patches plus augmentation give a near-endless stream of distinct inputs to a network with millions of parameters. Stage 2 then shows which component is responsible.

### Stage 3: ablation

`stage3_ablation.sh` runs these in priority order; `PRIORITY=1` runs only the first five. Each changes one thing against `baseline_seed42`.

| Priority | Ablation | Question |
|---|---|---|
| 1 | No skip connections | Are skips really needed? This links to the course hint |
| 1 | No augmentation | How much does augmentation buy? |
| 1 | Left-right flip only | Do anatomically impossible flips (front-back, up-down) hurt? |
| 1 | Loss: CE only, Dice only, Dice+CE | Which loss handles 5% foreground best? |
| 2 | Dropout 0 and 0.3 | Effect on accuracy and on MC-dropout uncertainty |
| 2 | Residual units 2 → 0 | Are residual blocks needed? |
| 2 | Depth 5 → 3 levels | Is a deep encoder needed for small volumes? Note this also cuts parameters from 5.7M to 0.34M |
| 3 | Batch norm instead of instance norm | Only if time allows |

### Stage 4: uncertainty and calibration

`stage4_uncertainty.sh` works on the validation set:

- Compares a single softmax pass, MC dropout, and a deep ensemble of the three baseline seeds, which costs no extra training.
- Measures how well each method's entropy flags error voxels, as an ROC curve and AUROC. This gives the ROC the course asks for.
- Keeps the case-level check: correlation of mean uncertainty with Dice.
- Fits temperature scaling on validation and reports foreground calibration error before and after.

### Stage 5: final test evaluation

`stage5_test.sh` is the only script that touches the test set. It evaluates every finished model, reports train, validation and test performance for the baseline, and reruns stage 4 on test. Report Dice, precision, recall, HD95, the confusion matrix, and qualitative examples including the worst case.

## 5. Schedule

| Days | Work |
|---|---|
| Mon Nov 2 | Smoke test on the GPU machine, then stage 1a |
| Tue Nov 3 | Stages 1b and 1c. Start stage 2 overnight |
| Wed Nov 4 – Thu Nov 5 | Stages 2 and 3, queued back to back |
| Fri Nov 6 | Stages 4 and 5 |
| Sat Nov 7 – Mon Nov 9 | Slides and two timed rehearsals |
| Tue Nov 10 | Send slides |
| Fri Nov 13 | Present |

If you fall behind, run stage 3 with `PRIORITY=1` and skip the data-size runs in stage 2. Never cut the loss plot, the overfitting study or the test metrics.

## 6. Slide outline for 12 minutes

1. Title and the question you answer
2. Task and data: MSD Task04, 260 MRI volumes, 2 structures, about 5% foreground, 70/15/15 split
3. Network: 3D U-Net diagram, parameter count, input and output shapes
4. Training setup: loss, optimiser, schedule, regularisation, augmentation
5. Baseline v1 to v2: what was wrong and what fixing it gained
6. Training and validation loss and Dice curves
7. Hyperparameter tuning table
8. Overfitting study curves and the component that prevents overfitting
9. Ablation table with one-line takeaways
10. Test results: metrics table, confusion matrix, good and bad examples
11. Uncertainty: error-detection ROC, maps, calibration before and after temperature scaling
12. Conclusions and limitations

Keep extra figures in backup slides for questions.

## 7. Literature

| Paper | Use it for |
|---|---|
| Ronneberger, Fischer, Brox. U-Net. MICCAI 2015 | Skip connections, encoder–decoder design |
| Çiçek et al. 3D U-Net. MICCAI 2016 | 3D extension |
| Isensee et al. nnU-Net. Nature Methods 2021 | Strong reference numbers and training recipe for this exact task |
| Antonelli et al. The Medical Segmentation Decathlon. Nature Communications 2022 | Dataset description and citation |
| Gal & Ghahramani. Dropout as a Bayesian Approximation. ICML 2016 | MC dropout |
| Lakshminarayanan et al. Deep Ensembles. NeurIPS 2017 | Ensemble baseline |
| Guo et al. On Calibration of Modern Neural Networks. ICML 2017 | ECE and temperature scaling |
| Mehrtash et al. Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation. IEEE TMI 2020 | Calibration in medical segmentation |
| Jungo & Reyes. Assessing Reliability and Challenges of Uncertainty Estimations for Medical Image Segmentation. MICCAI 2019 | How to evaluate uncertainty against errors |

## 8. Questions to prepare for

- Why Dice plus cross-entropy instead of either alone?
- Why instance norm instead of batch norm with batch size 2?
- Why is specificity about 0.99 even for a poor model?
- How did you make sure the test set never influenced a decision?
- Why did MC dropout raise Dice slightly? It averages 20 predictions, which acts like a small ensemble.
- Is MC dropout really Bayesian? It is an approximation, and it is often less reliable than ensembles.
- Why 3D instead of 2D slices?
- How far is your Dice from nnU-Net, and why?
