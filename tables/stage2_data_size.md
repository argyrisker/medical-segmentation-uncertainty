## stage2_data_size

| run | n_train | params | epochs | best_epoch | best_val_dice | elapsed_min | train_dice_at_best | final_train_dice | final_val_dice | final_gap | min_val_loss_epoch | val_dice_diff_vs_ref | val_wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_aug_dropout | 10 | 5733475 | 200 | 53 | 0.8310 | 12.3151 | 0.9443 | 0.9718 | 0.8236 | 0.1482 | 14 | -0.0453 | <0.0001 |
| baseline_frac0.25 | 46 | 5733475 | 200 | 75 | 0.8649 | 13.0127 | 0.9201 | 0.9424 | 0.8594 | 0.0830 | 75 | -0.0114 | 0.0043 |
| baseline_frac0.5 | 91 | 5733475 | 200 | 77 | 0.8696 | 13.0213 | 0.8967 | 0.9293 | 0.8651 | 0.0642 | 87 | -0.0067 | 0.2091 |
| baseline_seed42 | 182 | 5733475 | 200 | 96 | 0.8763 | 13.0122 | 0.9039 | 0.9163 | 0.8732 | 0.0430 | 96 |  |  |
