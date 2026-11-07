## stage1b_lr

| run | n_train | params | epochs | best_epoch | best_val_dice | elapsed_min | train_dice_at_best | final_train_dice | final_val_dice | final_gap | min_val_loss_epoch | val_dice_diff_vs_ref | val_wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1b_lr1e-4 | 182 | 5733475 | 200 | 101 | 0.8683 | 13.0510 | 0.9021 | 0.9118 | 0.8661 | 0.0458 | 101 | -0.0080 | 0.0006 |
| s1a_patchfull | 182 | 5733475 | 200 | 96 | 0.8763 | 13.0167 | 0.9039 | 0.9163 | 0.8732 | 0.0430 | 96 |  |  |
| s1b_lr1e-3 | 182 | 5733475 | 200 | 181 | 0.8758 | 13.0554 | 0.9198 | 0.9209 | 0.8748 | 0.0462 | 83 | -0.0005 | 0.9945 |
