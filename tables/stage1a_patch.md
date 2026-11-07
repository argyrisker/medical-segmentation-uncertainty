## stage1a_patch

| run | n_train | params | epochs | best_epoch | best_val_dice | elapsed_min | train_dice_at_best | final_train_dice | final_val_dice | final_gap | min_val_loss_epoch | val_dice_diff_vs_ref | val_wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_legacy | 182 | 4807482 | 100 | 100 | 0.6839 | 6.7516 | 0.6894 | 0.6894 | 0.6839 | 0.0055 | 100 |  |  |
| s1a_patch32 | 182 | 5733475 | 200 | 167 | 0.8867 | 14.4069 | 0.9187 | 0.9199 | 0.8859 | 0.0340 | 166 | 0.2028 | <0.0001 |
| s1a_patchfull | 182 | 5733475 | 200 | 96 | 0.8763 | 13.0167 | 0.9039 | 0.9163 | 0.8732 | 0.0430 | 96 | 0.1924 | <0.0001 |
