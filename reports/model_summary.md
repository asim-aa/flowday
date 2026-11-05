| Estimator | MAE, all test days | MAE, first 7 days | MAE, eval period |
|---|---|---|---|
| trust | 23.2 [21.6, 24.9] | 22.9 [21.0, 24.7] | 23.1 [21.4, 24.9] |
| global_multiplier | 18.5 [17.3, 19.8] | 18.1 [16.8, 19.6] | 18.4 [17.2, 19.7] |
| type_median | 14.6 [13.6, 15.6] | 14.4 [13.3, 15.6] | 14.6 [13.6, 15.7] |
| gbm | 11.8 [11.1, 12.6] | 12.6 [11.6, 13.6] | 11.7 [10.9, 12.5] |
| gbm_q50 | 11.9 [11.1, 12.6] | 12.7 [11.8, 13.7] | 11.7 [10.9, 12.5] |
| gbm_q80 | 16.2 [15.5, 16.9] | 17.4 [16.6, 18.2] | 16.0 [15.1, 16.8] |
| oracle_median | 11.4 [10.6, 12.1] | 11.4 [10.6, 12.3] | 11.3 [10.5, 12.1] |

Shifted population (log-bias +0.30), models not retrained:

| Slice | MAE trust | MAE type-median | MAE GBM | q50 coverage (0.50) | q80 coverage (0.80) |
|---|---|---|---|---|---|
| cold_start_first_7_days | 44.5 | 29.0 | 19.5 | 0.35 | 0.68 |
| eval_period | 43.6 | 28.2 | 16.1 | 0.48 | 0.80 |
