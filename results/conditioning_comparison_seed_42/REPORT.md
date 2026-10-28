# Matched circuit training test

Both variants trained for 30 epochs on the same full Loan split and seed 42.
Checkpoint selection uses highest validation macro F1; the following scores are on the shared final real test set.

| Circuit | Checkpoint | Epoch | Accuracy | Macro F1 | AUROC | Average precision | Mean raw numeric KS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper + D(x) | last | 30 | 0.5790 | 0.4577 | 0.6963 | 0.2342 | 0.6067 |
| paper + D(x) | best | 1 | 0.7220 | 0.4579 | 0.4451 | 0.0877 | 0.8872 |
| paper + D(x,y) | last | 30 | 0.5990 | 0.4822 | 0.7328 | 0.3349 | 0.5973 |
| paper + D(x,y) | best | 10 | 0.5810 | 0.4787 | 0.6696 | 0.1932 | 0.5936 |

Real-trained reference: accuracy 0.9830, macro F1 0.9473, AUROC 0.9966, average precision 0.9778.
Majority-class test accuracy: 0.9040. One seed and a short budget are exploratory evidence, not proof of a reliable winner or a paper reproduction.

## Diversity checks

| Circuit | Checkpoint | Mean numeric std / real std | Mean minimum within-class std / real std | Mean category coverage |
| --- | --- | ---: | ---: | ---: |
| paper + D(x) | last | 0.00295 | 0.00122 | 0.3896 |
| paper + D(x) | best | 0.00165 | 0.00076 | 0.3893 |
| paper + D(x,y) | last | 0.00956 | 0.00178 | 0.3893 |
| paper + D(x,y) | best | 0.00703 | 0.00416 | 0.3896 |

Ratios near zero indicate collapse; a ratio near one alone does not establish a distribution match. Lower numeric KS is better. Category coverage is an unweighted average across categorical features.

[Validation trajectories](validation_comparison.png) · [Losses and quantum gradients](loss_comparison.png) · [Complete numbers](summary.csv)

Each source run contains test/real, test/last and test/best confusion matrices, ROC and precision–recall plots, predictions, synthetic rows and raw diagnostics. Runs:

- `results/conditioning_trial_30_control`
- `results/conditioning_trial_30_conditional`
