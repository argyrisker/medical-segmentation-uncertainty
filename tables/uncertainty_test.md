## uncertainty_test

| run_name | mode | split | n_passes | dice_mean | dice_std | hd95_mean | error_auroc_pooled | error_auroc_case_mean | ece_foreground | entropy_error_mean | entropy_correct_foreground_mean | dice_vs_entropy_pearson_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed42 | deterministic | test | 1 | 0.8792 | 0.0406 | 1.2956 | 0.8375 | 0.8524 | 0.1640 | 0.2591 | 0.0467 | -0.0219 |
| baseline_seed42 | mc_dropout | test | 20 | 0.8793 | 0.0392 | 1.3006 | 0.8245 | 0.8387 | 0.1449 | 0.3494 | 0.0800 | -0.0715 |
| baseline_ensemble | ensemble | test | 3 | 0.8821 | 0.0349 | 1.2547 | 0.8576 | 0.8670 | 0.1453 | 0.3258 | 0.0615 | -0.3102 |
