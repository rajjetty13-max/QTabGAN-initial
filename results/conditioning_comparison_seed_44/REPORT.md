# Matched circuit training test

Both variants trained for 30 epochs on the same full Loan split and seed 44.
Checkpoint selection uses highest validation macro F1; the following scores are on the shared final real test set.

| Circuit | Checkpoint | Epoch | Accuracy | Macro F1 | AUROC | Average precision | Mean raw numeric KS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper + D(x) | last | 30 | 0.7100 | 0.5502 | 0.7202 | 0.3531 | 0.5997 |
| paper + D(x) | best | 30 | 0.7100 | 0.5502 | 0.7202 | 0.3531 | 0.5997 |
| paper + D(x,y) | last | 30 | 0.5870 | 0.4718 | 0.7185 | 0.3252 | 0.6140 |
| paper + D(x,y) | best | 10 | 0.8760 | 0.6556 | 0.7397 | 0.3393 | 0.6651 |

Real-trained reference: accuracy 0.9830, macro F1 0.9473, AUROC 0.9902, average precision 0.9538.
Majority-class test accuracy: 0.9040. One seed and a short budget are exploratory evidence, not proof of a reliable winner or a paper reproduction.

## Diversity checks

| Circuit | Checkpoint | Mean numeric std / real std | Mean minimum within-class std / real std | Mean category coverage |
| --- | --- | ---: | ---: | ---: |
| paper + D(x) | last | 0.05268 | 0.00616 | 0.3896 |
| paper + D(x) | best | 0.05268 | 0.00616 | 0.3896 |
| paper + D(x,y) | last | 0.01249 | 0.00248 | 0.3893 |
| paper + D(x,y) | best | 0.03994 | 0.00685 | 0.3896 |

Ratios near zero indicate collapse; a ratio near one alone does not establish a distribution match. Lower numeric KS is better. Category coverage is an unweighted average across categorical features.

[Validation trajectories](validation_comparison.png) · [Losses and quantum gradients](loss_comparison.png) · [Complete numbers](summary.csv)

Each source run contains test/real, test/last and test/best confusion matrices, ROC and precision–recall plots, predictions, synthetic rows and raw diagnostics. Runs:

- `results/conditioning_trial_30_control_seed_44`
- `results/conditioning_trial_30_conditional_seed_44`
