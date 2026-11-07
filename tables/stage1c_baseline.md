## stage1c_baseline

| run | n_train | params | epochs | best_epoch | best_val_dice | elapsed_min | train_dice_at_best | final_train_dice | final_val_dice | final_gap | min_val_loss_epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed42 | 182 | 5733475 | 200 | 96 | 0.8763 | 13.0122 | 0.9039 | 0.9163 | 0.8732 | 0.0430 | 96 |
| baseline_seed43 | 182 | 5733475 | 200 | 117 | 0.8770 | 12.9964 | 0.9080 | 0.9177 | 0.8730 | 0.0447 | 117 |
| baseline_seed44 | 182 | 5733475 | 200 | 109 | 0.8774 | 13.0487 | 0.9032 | 0.9193 | 0.8718 | 0.0476 | 109 |

### Averaged over seeds

| experiment | best_val_dice_mean | best_val_dice_std | best_val_dice_count | final_gap_mean | final_gap_std | final_gap_count |
|---|---|---|---|---|---|---|
| baseline | 0.8769 | 0.0006 | 3 | 0.0451 | 0.0023 | 3 |
