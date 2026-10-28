# Matched circuit training test

Both variants trained for 30 epochs on the same full Loan split and seed 43.
Checkpoint selection uses highest validation macro F1; the following scores are on the shared final real test set.

| Circuit | Checkpoint | Epoch | Accuracy | Macro F1 | AUROC | Average precision | Mean raw numeric KS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper + D(x) | last | 30 | 0.6140 | 0.5021 | 0.7124 | 0.3299 | 0.6057 |
| paper + D(x) | best | 1 | 0.8970 | 0.5470 | 0.5648 | 0.1698 | 0.9142 |
| paper + D(x,y) | last | 30 | 0.4240 | 0.3735 | 0.6896 | 0.2170 | 0.6004 |
| paper + D(x,y) | best | 10 | 0.7650 | 0.5958 | 0.7624 | 0.3994 | 0.6246 |

Real-trained reference: accuracy 0.9780, macro F1 0.9293, AUROC 0.9879, average precision 0.9523.
Majority-class test accuracy: 0.9040. One seed and a short budget are exploratory evidence, not proof of a reliable winner or a paper reproduction.

## Diversity checks

| Circuit | Checkpoint | Mean numeric std / real std | Mean minimum within-class std / real std | Mean category coverage |
| --- | --- | ---: | ---: | ---: |
| paper + D(x) | last | 0.00231 | 0.00080 | 0.3893 |
| paper + D(x) | best | 0.00259 | 0.00067 | 0.4729 |
| paper + D(x,y) | last | 0.00472 | 0.00231 | 0.3893 |
| paper + D(x,y) | best | 0.02941 | 0.00506 | 0.3893 |

Ratios near zero indicate collapse; a ratio near one alone does not establish a distribution match. Lower numeric KS is better. Category coverage is an unweighted average across categorical features.

[Validation trajectories](validation_comparison.png) · [Losses and quantum gradients](loss_comparison.png) · [Complete numbers](summary.csv)

Each source run contains test/real, test/last and test/best confusion matrices, ROC and precision–recall plots, predictions, synthetic rows and raw diagnostics. Runs:

- `results/conditioning_trial_30_control_seed_43`
- `results/conditioning_trial_30_conditional_seed_43`
