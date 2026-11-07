## stage2_overfit

| run | n_train | params | epochs | best_epoch | best_val_dice | elapsed_min | train_dice_at_best | final_train_dice | final_val_dice | final_gap | min_val_loss_epoch | val_dice_diff_vs_ref | val_wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_noreg | 10 | 5733475 | 200 | 13 | 0.8350 | 11.7865 | 0.9692 | 1.0000 | 0.8228 | 0.1772 | 5 |  |  |
| s2_aug | 10 | 5733475 | 200 | 37 | 0.8260 | 11.8959 | 0.9583 | 0.9997 | 0.8096 | 0.1901 | 17 | -0.0090 | 0.0640 |
| s2_aug_dropout | 10 | 5733475 | 200 | 53 | 0.8310 | 12.3151 | 0.9443 | 0.9718 | 0.8236 | 0.1482 | 14 | -0.0039 | 0.9395 |
| s2_aug_dropout_wd | 10 | 5733475 | 200 | 52 | 0.8321 | 12.2685 | 0.9477 | 0.9720 | 0.8241 | 0.1479 | 19 | -0.0029 | 0.9395 |
