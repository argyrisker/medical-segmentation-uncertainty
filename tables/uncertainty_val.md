## uncertainty_val

| run_name | mode | split | n_passes | dice_mean | dice_std | hd95_mean | error_auroc_pooled | error_auroc_case_mean | ece_foreground | entropy_error_mean | entropy_correct_foreground_mean | dice_vs_entropy_pearson_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_seed42 | deterministic | val | 1 | 0.8763 | 0.0410 | 1.4682 | 0.8356 | 0.8479 | 0.1648 | 0.2569 | 0.0461 | -0.3180 |
| baseline_seed42 | mc_dropout | val | 20 | 0.8761 | 0.0409 | 1.4590 | 0.8236 | 0.8354 | 0.1450 | 0.3516 | 0.0810 | -0.4511 |
| baseline_ensemble | ensemble | val | 3 | 0.8799 | 0.0374 | 1.4263 | 0.8589 | 0.8675 | 0.1442 | 0.3361 | 0.0612 | -0.4388 |
