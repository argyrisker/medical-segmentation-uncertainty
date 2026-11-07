# Slide content: results from the HH GPU run

Numbers come from the full run that finished on Nov 7 (200 epochs per run,
RTX 4000 Ada, about 13 minutes per run). Validation numbers were used for every
decision; test numbers are reported once, at the end. "±" is a standard deviation.
Every number is filled in.

---

## 1. Title

**What matters in a 3D U-Net for hippocampus segmentation, and does it know when it is wrong?**

Your name, course, date.

---

## 2. Task and data

- Medical Segmentation Decathlon, Task04 Hippocampus: 260 labelled MRI scans, about 35×50×36 voxels at 1 mm.
- Three classes: background, anterior and posterior hippocampus. Foreground is only about 5% of each scan.
- Fixed split: 182 training, 39 validation, 39 test scans. The official test labels are hidden.

Figures already in the repo: `reports/figures/report/hippocampus_279_three_plane_overlay.png`,
`reports/figures/report/foreground_fraction_histogram.png`.

**Say:** a small structure with strong class imbalance; every decision was made on validation and the test set was used once.

---

## 3. Network

- Own 3D U-Net, five levels with 16, 32, 64, 128 and 256 channels, 5.7 million parameters.
- Each block: two 3×3×3 convolutions, instance norm, LeakyReLU, dropout 0.1, and a residual shortcut.
- Down: 2×2×2 max pooling. Up: transposed convolution, then concatenation with the encoder features (skip connection).
- Head: 1×1×1 convolution to 3 classes. Input: the whole scan, padded to 48×64×48.

Figure: a U-Net diagram like the one on the course project slide.

---

## 4. Training setup

| Item | Choice |
|---|---|
| Loss | Dice + cross-entropy |
| Optimiser | AdamW, learning rate 3e-4, weight decay 1e-5 |
| Schedule | Cosine decay over 200 epochs |
| Batch | 4 whole scans |
| Regularisation | Dropout 0.1, flips on all axes, intensity shift, best-validation checkpoint |
| Time | About 13 minutes per run |

---

## 5. Fixing the original pipeline

| | v1 (original) | v2 baseline, 3 seeds |
|---|---|---|
| Validation Dice | 0.684 | 0.877 ± 0.001 |
| Test Dice | 0.700 | 0.878 ± 0.002 |
| Test HD95 | 16.7 mm | 1.30 mm |
| Training Dice, saved model | 0.689 | 0.905 |

Figure: `reports/figures/curves/stage1a_patch_val_dice.png`

v1 was also worst on the anterior part (Dice 0.68, HD95 13.8 mm in the original
report); v2 now does slightly better on anterior than posterior.

**Say:** v1 scored the same on its own training scans as on validation, so it was not
overfitting. It failed when predicting whole scans: it trained on 32³ cubes but
predicted with 64³ windows that were about three quarters zero padding, and it was
still improving when training stopped. v2 changed several things at once (matching
windows, own network, schedule, more epochs), so the gain belongs to the bundle.

---

## 6. Learning curves of the final model

Figure: `reports/figures/curves/baseline_seed42_curves.png`

- Validation loss is lowest at epoch 96, where validation Dice is also best.
- Training keeps improving after that: final training Dice 0.916 against validation 0.873, a gap of 0.04.
- The saved model is the epoch-96 checkpoint.

**Say:** mild overfitting late in training; choosing the best validation checkpoint acts as early stopping.

---

## 7. Hyperparameter tuning (validation)

| Setting | Best validation Dice |
|---|---|
| 32³ patches, sliding window | 0.887 |
| Whole scan, 48×64×48 | 0.876 |
| Learning rate 1e-4 | 0.868 |
| Learning rate 3e-4 | 0.876 |
| Learning rate 1e-3 | 0.876 |

Figure: `reports/figures/curves/stage1b_lr_val_dice.png`

**Say:** 3e-4 and 1e-3 tie; 1e-4 learns too slowly. Small patches scored 0.011
higher, but they also took twice as many optimiser steps per epoch (91 against 46),
so the comparison is not fully fair. The ablations use the whole-scan model as the reference.
The small-patch model was also better on test: Dice 0.886 against 0.879, HD95 1.25 against 1.30 mm.

---

## 8. Overfitting study: 10 training scans

| Regularisation | Final train Dice | Best validation Dice | Final gap |
|---|---|---|---|
| None | 1.000 | 0.835 | 0.18 |
| + augmentation | 1.000 | 0.826 | 0.19 |
| + dropout 0.1 | 0.972 | 0.831 | 0.15 |
| + weight decay 1e-2 | 0.972 | 0.832 | 0.15 |

Figures: `reports/figures/curves/stage2_overfit_val_loss.png`, `reports/figures/curves/stage2_overfit_train_eval_dice.png`

**Say:** without regularisation the network memorises the 10 scans perfectly, and
validation loss is lowest at epoch 5 and rises after. Dropout stops perfect
memorisation; augmentation and weight decay change almost nothing; none of them
improves validation.

---

## 9. What really prevents overfitting: more data

| Training scans | Best validation Dice | Final gap |
|---|---|---|
| 10 | 0.831 | 0.15 |
| 46 | 0.865 | 0.08 |
| 91 | 0.870 | 0.06 |
| 182 | 0.876 | 0.04 |

Figure: `reports/figures/curves/stage2_data_size_val_dice.png`

**Say:** every doubling of the data shrinks the gap. With all 182 scans the model
barely overfits, which is why the full-data model could not be made to overfit.

---

## 10. Ablation: one change at a time

| Change | Validation ΔDice | p | Test ΔDice | p |
|---|---|---|---|---|
| Remove skip connections | −0.048 | <0.0001 | −0.041 | <0.0001 |
| No augmentation | +0.006 | 0.04 | +0.006 | 0.02 |
| Left-right flip only | +0.007 | 0.29 | +0.000 | 0.90 |
| Cross-entropy only | −0.004 | 0.16 | −0.000 | 0.97 |
| Dice only | +0.003 | 0.64 | +0.004 | 0.22 |

Seed-to-seed spread of the baseline: ±0.001 on validation, ±0.002 on test.
p-values: paired Wilcoxon test over the 39 scans.

Figure: `reports/figures/curves/stage3_ablation_val_dice.png`

**Say:** skip connections are the only part that clearly matters. Without them the
decoder cannot recover boundaries: test HD95 rises from 1.30 to 1.58 mm, and even
training Dice drops (0.888). Augmentation and the loss choice move Dice by less than
0.01, and with five tests (p below 0.01 needed) none is significant. Without
augmentation the train-validation gap doubles (0.04 to 0.08), yet validation does
not drop: it reduces overfitting without improving results here.

---

## 11. Test results of the final model

| Metric | Train (182) | Validation (39) | Test (39) |
|---|---|---|---|
| Dice, anterior | 0.905 | 0.884 | 0.885 |
| Dice, posterior | 0.890 | 0.869 | 0.874 |
| Dice, mean | 0.898 ± 0.027 | 0.876 ± 0.041 | 0.879 ± 0.041 |
| Precision | 0.887 | 0.865 | 0.864 |
| Recall (sensitivity) | 0.911 | 0.892 | 0.898 |
| HD95 | 1.10 mm | 1.47 mm | 1.30 mm |

Across three training seeds the test Dice is 0.878 ± 0.002.

- Confusion matrix: `outputs/metrics/baseline_seed42/test_confusion_matrix.png`.
- Good and bad examples: `outputs/predictions/baseline_seed42/`.
- Context: nnU-Net reports about 0.89–0.90 on the official MSD test set. Different scans, so not directly comparable.

**Say:** test is as good as validation, so model selection did not overfit the
validation set. Recall is higher than precision: the model draws the boundary
slightly too wide. Specificity is about 0.99 for any model because background
dominates, so it is not the headline metric.

---

## 12. Does the model know when it is wrong?

| Method | Dice | Error AUROC | Foreground ECE |
|---|---|---|---|
| Softmax, one pass | 0.879 | 0.838 | 0.164 |
| MC dropout, 20 passes | 0.879 | 0.825 | 0.145 |
| Ensemble of 3 seeds | 0.882 | 0.858 | 0.145 |

Figures: `reports/figures/uncertainty/uncertainty_test_roc.png`,
`reports/figures/baseline_seed42/test_reliability_foreground.png`,
maps in `reports/figures/baseline_seed42/mc_dropout/`.
Temperature scaling, fitted on validation and applied to test:

| | Before | After (T = 3.3) |
|---|---|---|
| Foreground ECE | 0.164 | 0.051 |
| Foreground NLL | 0.85 | 0.43 |
| All-voxel ECE | 0.009 | 0.018 |

**Say:** uncertainty is about five times higher on wrong voxels than on correct
foreground voxels. An AUROC of 0.84 means an error voxel usually has higher
uncertainty than a correct one. MC dropout is no better than the plain softmax at
finding errors; the three-model ensemble is best on every measure. Per scan, only the
ensemble's uncertainty tracks Dice, and weakly (r = −0.31). The model is
overconfident on the hippocampus; dividing its logits by 3.3 cuts the foreground
calibration error by a factor of three without changing any prediction. The
all-voxel error is tiny either way, because easy background dominates it.

---

## 13. Conclusions and limitations

- Fixing the pipeline raised test Dice from 0.70 to 0.88 and cut HD95 from 16.7 to 1.3 mm.
- Skip connections are essential; augmentation and loss choice matter little on this data.
- More data is the strongest regulariser.
- A small ensemble beats MC dropout for uncertainty, and temperature scaling fixes most of the foreground overconfidence.

Limitations: one dataset and an internal split with 39 test scans; ablations use one
seed (the seed spread is small); the v1-to-v2 gain bundles several changes; the patch
comparison also differs in optimiser steps.

---

## Likely questions

- **Why did v1 fail?** Training Dice equalled validation Dice (0.69), so it was not overfitting. Its prediction windows were mostly padding, and it was still improving at epoch 100.
- **Why is specificity so high?** About 95% of voxels are background, so almost any model gets it right.
- **How did you avoid test leakage?** All choices used validation; the test set is evaluated by one final script.
- **Why is the seed spread ±0.001 but the per-scan spread ±0.04?** The first is how much the model changes between training runs; the second is how much scans differ in difficulty.
- **Is MC dropout Bayesian?** It is an approximation, and here it did not beat a single softmax pass.
- **Why is precision lower than recall?** The model draws the boundary slightly too wide, so it includes a few extra voxels.
- **Why fit the temperature on foreground voxels only?** That is where the overconfidence is; fitting on all voxels would be dominated by easy background.
- **Why does augmentation not help?** 182 scans already cover the anatomical variation; augmentation narrows the train-validation gap but does not raise validation Dice.

---

## Optional: picture of the worst test scan

The example figures cover the first 8 test scans. To also have the worst one,
save a figure for every test scan and list the three lowest:

```bash
python -m src.evaluate --checkpoint outputs/checkpoints/baseline_seed42/best_model.pt --split test --num-vis 39
python -c "import pandas as pd; d = pd.read_csv('outputs/metrics/baseline_seed42/test_metrics.csv'); print(d.nsmallest(3, 'dice_mean')[['case_id', 'dice_mean', 'hd95_mean']])"
```

The figures are in `outputs/predictions/baseline_seed42/`, named after the scan.
