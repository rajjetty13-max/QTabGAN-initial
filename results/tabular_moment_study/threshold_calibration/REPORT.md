# Validation-only decision-threshold calibration

Every model keeps the checkpoint already selected by its original validation macro F1. The generator candidate remains **moment_matched**, chosen in the original validation-only model comparison. Calibration does not reselect checkpoints or rank model variants.

For each group and seed, the decision threshold maximizes macro F1 using only predictions from that checkpoint’s validation epoch. The grid is `np.linspace(0, 1, 101)`. Ties choose the threshold closest to 0.5, then the larger threshold. A row is positive only when `positive_probability > threshold`; this preserves the random forest’s negative decision on a 0.5 tie.

Each threshold was applied once to the saved predictions for that checkpoint’s test set. No test score, label or probability was used to choose thresholds or candidate models. All validation/test saved labels were verified to equal the original `probability > 0.5` rule, paired split rows and truth were checked, and original test metrics were reproduced from predictions.

## Test results

Means ± sample SD across seeds 42, 43 and 44. Original means use threshold 0.5. AUROC and AP remain exactly unchanged because thresholding does not change the probabilities.

| Variant | Decision rule | Accuracy | Macro F1 | Minority recall | AUROC | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | original | 0.7697 ± 0.1312 | 0.6013 ± 0.1000 | 0.5486 ± 0.1853 | 0.7923 ± 0.1012 | 0.3563 ± 0.1838 |
| control | calibrated | 0.8390 ± 0.1006 | 0.6409 ± 0.0959 | 0.4792 ± 0.2345 | 0.7923 ± 0.1012 | 0.3563 ± 0.1838 |
| typed_output | original | 0.8327 ± 0.0332 | 0.6164 ± 0.0610 | 0.4444 ± 0.2191 | 0.7621 ± 0.0610 | 0.3392 ± 0.1731 |
| typed_output | calibrated | 0.8303 ± 0.0949 | 0.6500 ± 0.0765 | 0.5347 ± 0.0789 | 0.7621 ± 0.0610 | 0.3392 ± 0.1731 |
| moment_matched | original | 0.8520 ± 0.0183 | 0.6971 ± 0.0068 | 0.7153 ± 0.1104 | 0.8764 ± 0.0136 | 0.4093 ± 0.0563 |
| moment_matched | calibrated | 0.8600 ± 0.0095 | 0.6978 ± 0.0081 | 0.6632 ± 0.0335 | 0.8764 ± 0.0136 | 0.4093 ± 0.0563 |

## Frozen checkpoints and thresholds

| Variant | Seed | Original best epoch | Threshold | Validation macro F1 before | Validation macro F1 after |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 42 | 20 | 0.46 | 0.6571 | 0.7476 |
| control | 43 | 20 | 0.60 | 0.5402 | 0.5604 |
| control | 44 | 10 | 0.72 | 0.5234 | 0.6371 |
| typed_output | 42 | 1 | 0.48 | 0.6654 | 0.6946 |
| typed_output | 43 | 10 | 0.59 | 0.6269 | 0.7366 |
| typed_output | 44 | 30 | 0.36 | 0.5485 | 0.5735 |
| moment_matched | 42 | 20 | 0.51 | 0.7204 | 0.7211 |
| moment_matched | 43 | 20 | 0.49 | 0.7460 | 0.7483 |
| moment_matched | 44 | 30 | 0.57 | 0.6698 | 0.7242 |

## Scope

This is a separate evaluation extension, not an improvement to generator training, data fidelity or quantum circuitry. Calibration can improve an older model too; all three groups receive the same procedure. The moment objective and typed output remain distinct model extensions to the paper. No model was retrained, and no raw report or prediction file was overwritten.

These validation/test splits have been examined during earlier exploratory work. This additional evaluation is exploratory and does not establish independent confirmation or statistical significance. Macro-F1 calibration may trade accuracy against minority recall. The 90.4% majority-class accuracy reference remains relevant.

[Per-seed validation and test scores](scores.csv) · [Selected thresholds](selected_thresholds.csv) · [Full validation grid](threshold_grid.csv) · [Calibrated predictions](predictions.csv) · [Input hashes](input_hashes.csv)
