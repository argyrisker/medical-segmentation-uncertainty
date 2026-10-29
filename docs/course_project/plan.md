# Course project plan: 3D hippocampus segmentation

Course: Introduction to Deep Learning 2026, final project option 3, "bring your own data/task".

| Date | What |
|---|---|
| Now | Email the one-page proposal in `proposal_draft.md` and wait for approval before starting |
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
3. **Foreground crop after z-scoring.** In `src/transforms.py`, `CropForegroundd` runs after `NormalizeIntensityd`. Its default keeps voxels greater than zero, and after z-scoring every below-mean tissue voxel is negative. The crop box is then computed from bright voxels only and can clip labelled voxels. Fix: crop before normalising, or drop the crop because MSD volumes are already cropped. Check the effect in the notebook by counting label voxels before and after.
4. **Anterior–posterior flips.** Labels 1 and 2 are anterior and posterior hippocampus. After RAS reorientation, axis 1 is the anterior–posterior axis, and the pipeline flips it and rotates in the axial plane. This produces anatomically impossible training images. nnU-Net mirrors all axes and still does well, so treat this as an ablation, not a bug.
5. **No validation loss and no train Dice.** Without these you cannot draw the required loss plot or diagnose overfitting.

## 4. Experiment plan

Ground rules:

- Make every tuning and ablation decision on the validation set. Touch the test set once, for the final models.
- Keep the split and seed fixed across runs. Use 3 seeds for the final baseline. The test set has only 39 cases, so report mean ± SD and a paired Wilcoxon test for key comparisons.
- Log per epoch: train loss, val loss, train Dice on a fixed subset of about 20 training cases, val Dice, learning rate.
- Set `cache_rate: 1.0`. The whole dataset fits in memory easily.
- Time the first full run, then budget the rest. Queue runs overnight.

### Stage 0: code changes

| File | Change |
|---|---|
| `src/train.py` | Validation loss on full val volumes, train Dice on a fixed subset, LR logging, cosine schedule option, `--seed` override, write history every epoch |
| new `src/plot_curves.py` | Learning curves from `training_history.csv`, several runs on one figure |
| `src/models.py` | Custom 3D U-Net with a `use_skips` switch, because MONAI's `UNet` always has skips. Expose depth, residual units, norm type |
| `src/transforms.py` | Crop before normalising. Augmentation level: none, no A–P flip, full |
| `src/metrics.py` | Precision and a 3×3 voxel confusion matrix |
| `src/infer_uncertainty.py` | Deep-ensemble mode. Voxel-level AUROC of uncertainty for detecting errors |
| `src/calibration.py` | Temperature scaling fitted on the validation set |
| `configs/experiments/` | One small YAML per run, so every slide number maps to a config |

### Stage 1: fixed baseline and tuning

| Run | Change | Purpose |
|---|---|---|
| v1 | Current config, with new logging | Reference learning curves |
| v2-patch32 | Fixes 1–3, patch = roi = 32³ | Does fixing the mismatch help? |
| v2-patchfull | Fixes 1–3, patch = roi = 48×64×48 | Whole-volume context |
| lr sweep | lr in {1e-4, 3e-4, 1e-3}, cosine decay, 200 epochs | Pick the learning rate |
| batch | batch size 2 vs 4 | Only if time allows |

The winner on validation Dice becomes the baseline for every later stage.

### Stage 2: overfitting study

1. Train on 10 cases with no augmentation, dropout 0 and weight decay 0, for 300 epochs. Expect train loss near zero and validation loss rising.
2. Add back augmentation, then dropout, then weight decay, one per run. Plot the train–validation gap for each.
3. Optional: reuse the existing low-data script for 10, 25, 50 and 100% of training data with full regularisation.

If the full-data model will not overfit, say why: random patches plus augmentation give a near-endless stream of distinct inputs to a network with millions of parameters. Stage 2 then shows which component is responsible.

### Stage 3: ablation

All runs change one thing against the Stage 1 baseline. Priority 1 runs first.

| Priority | Ablation | Question |
|---|---|---|
| 1 | No skip connections | Are skips really needed? This links to the course hint |
| 1 | No augmentation | How much does augmentation buy? |
| 1 | No A–P flip or rotation | Do anatomically impossible flips hurt? |
| 1 | Loss: CE only, Dice only, Dice+CE | Which loss handles 5% foreground best? |
| 2 | Dropout 0 and 0.3 | Effect on accuracy and on MC-dropout uncertainty |
| 2 | Residual units 2 → 0 | Are residual blocks needed? |
| 2 | Depth 5 → 3 levels | Is a deep encoder needed for small volumes? |
| 3 | Batch norm instead of instance norm | Only if time allows |

### Stage 4: uncertainty and calibration

- Compare MC dropout with a deep ensemble built from the 3 baseline seeds, which costs nothing extra.
- Measure voxel-level AUROC of entropy for detecting errors. This gives the ROC curve the course asks for.
- Keep the case-level check: correlation of mean uncertainty with Dice.
- Fit temperature scaling on validation and report foreground ECE before and after.

### Stage 5: final test evaluation

Run the chosen models on the test set once. Report Dice, precision, recall, HD95, the confusion matrix, and qualitative examples including the worst case.

## 5. Schedule

| Days | Work |
|---|---|
| Thu Oct 29 – Fri Oct 30 | Send proposal. Stage 0 code. Smoke-test with `scripts/run_fast_dev.sh` |
| Sat Oct 31 – Mon Nov 2 | Stage 1 |
| Tue Nov 3 – Thu Nov 5 | Stages 2 and 3 |
| Fri Nov 6 – Sat Nov 7 | Stages 4 and 5 |
| Sun Nov 8 – Mon Nov 9 | Slides and two timed rehearsals |
| Tue Nov 10 | Send slides |
| Fri Nov 13 | Present |

If you fall behind, cut priority-2 and priority-3 ablations and the batch-size sweep first. Never cut the loss plot, the overfitting study or the test metrics.

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
