<!-- Draft one-page proposal. Replace the bracketed fields, then paste into the email. -->

**To:** mahmoud.Rahat@hh.se
**Subject:** Final project proposal (own data): 3D hippocampus segmentation in MRI – [Your name]

Dear Mahmoud,

I would like to do the final project with my own task and data. My one-page plan is below.

**Title.** What matters in a 3D U-Net for hippocampus segmentation, and does the network know when it is wrong?

**Motivation.** Hippocampus volume is an imaging biomarker in neurological and psychiatric disease, and manual segmentation in MRI is slow. The structure is small, about 5% of each scan, which makes it a hard segmentation problem with strong class imbalance.

**Data.** Medical Segmentation Decathlon, Task04 Hippocampus (Antonelli et al., 2022). It contains 260 labelled MRI volumes of about 35×50×36 voxels at 1 mm spacing, with three classes: background, anterior and posterior hippocampus. The data is public under CC BY-SA 4.0. The official test labels are not public, so I use a fixed internal split of 182 training, 39 validation and 39 test cases. The test set is used only once, for final evaluation.

**Network.** A 3D U-Net: a convolutional encoder–decoder with skip connections, residual units, instance normalisation and dropout. Implemented in PyTorch with MONAI, trained with a Dice plus cross-entropy loss and the AdamW optimiser.

**Experiments.**
1. Baseline and hyperparameter tuning on the validation set: learning rate and schedule, patch size and batch size, with training and validation loss curves over 100 or more epochs.
2. Overfitting study: train on a small subset without regularisation, then add augmentation, dropout and weight decay back one at a time to show which one prevents overfitting.
3. Ablation study: remove skip connections, residual units, encoder depth, dropout and augmentation, and compare loss functions.
4. Uncertainty: compare Monte Carlo dropout with a small deep ensemble, test whether uncertainty detects segmentation errors, and fix overconfidence with temperature scaling.

**Evaluation.** Dice (equal to F1), precision, recall, 95th-percentile Hausdorff distance and a voxel confusion matrix on the test set. For uncertainty, ROC/AUROC for error detection and expected calibration error. Results are reported as mean ± SD over test cases.

**Prior work.** This builds on a pipeline I wrote earlier: a 3D U-Net baseline with Monte Carlo dropout, which reached a test Dice of 0.72. New for this course are fixes to the training setup, learning curves, tuning, the overfitting and ablation studies, and the ensemble and calibration comparison.

**Compute and timeline.** Training runs on the HH GPU. Week one covers fixes, baseline and tuning. Week two covers overfitting, ablation and uncertainty. Slides will be sent by November 10.

**Key references.** Ronneberger et al., U-Net, MICCAI 2015. Çiçek et al., 3D U-Net, MICCAI 2016. Isensee et al., nnU-Net, Nature Methods 2021. Antonelli et al., The Medical Segmentation Decathlon, Nature Communications 2022. Gal & Ghahramani, Dropout as a Bayesian Approximation, ICML 2016.

Best regards,
[Your name], [programme / student ID]
