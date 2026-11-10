# Matched circuit training test

Both variants trained for 30 epochs on the same full Loan split and seed 42.
Checkpoint selection uses highest validation macro F1; the following scores are on the shared final real test set.

| Circuit | Checkpoint | Epoch | Accuracy | Macro F1 | AUROC | Average precision | Mean raw numeric KS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paper | last | 30 | 0.6380 | 0.5052 | 0.7413 | 0.3536 | 0.6036 |
| paper | best | 30 | 0.6380 | 0.5052 | 0.7413 | 0.3536 | 0.6036 |
| reuploading | last | 30 | 0.4250 | 0.3765 | 0.7009 | 0.2455 | 0.5924 |
| reuploading | best | 20 | 0.5410 | 0.4592 | 0.7377 | 0.2895 | 0.6134 |

Real-trained reference: accuracy 0.9830, macro F1 0.9473, AUROC 0.9966, average precision 0.9778.
Majority-class test accuracy: 0.9040. One seed and a short budget are exploratory evidence, not proof of a reliable winner or a paper reproduction.

## Diversity checks

| Circuit | Checkpoint | Mean numeric std / real std | Mean minimum within-class std / real std | Mean category coverage |
| --- | --- | ---: | ---: | ---: |
| paper | last | 0.01643 | 0.00207 | 0.3896 |
| paper | best | 0.01643 | 0.00207 | 0.3896 |
| reuploading | last | 0.00403 | 0.00202 | 0.3893 |
| reuploading | best | 0.00711 | 0.00422 | 0.3896 |

Ratios near zero indicate collapse; a ratio near one alone does not establish a distribution match. Lower numeric KS is better. Category coverage is an unweighted average across categorical features.

[Validation trajectories](validation_comparison.png) · [Losses and quantum gradients](loss_comparison.png) · [Complete numbers](summary.csv)

Each source run contains test/real, test/last and test/best confusion matrices, ROC and precision–recall plots, predictions, synthetic rows and raw diagnostics. Runs:

- `results/circuit_trial_30_paper`
- `results/circuit_trial_30_reuploading`

## Interpretation of this run

The new circuit did not improve downstream predictive utility in this screening test. With checkpoints chosen by validation macro F1, the original achieved test AUROC 0.7413, macro F1 0.5052 and average precision 0.3536; the alternative achieved 0.7377, 0.4592 and 0.2895. The AUROC difference is small and is not evidence of a reliable ranking across seeds. The alternative also performed worse at the shared final epoch 30.

Both generators collapsed. Mean minimum within-class numeric standard-deviation ratios were approximately 0.21% for the selected original and 0.42% for the selected alternative. These are averages of the smaller class-specific ratio per numeric feature, not averages over all class-feature pairs. Mean coverage across categorical features was about 39% for both; large raw numeric KS values also show poor distribution matches.

Removing inactive quantum parameters and reducing logical depth was a real structural improvement, but did not resolve the GAN's data-quality problem. Keep the original as the performance baseline. Before spending on longer training or hardware, investigate the classical mapper's dependence on random input and the discriminator's inability to check label/feature consistency. Those are hypotheses for a separate controlled experiment, not causes established by this comparison.

The unit-test suite passed all 23 tests before the comparison was reported. This experiment used analytic local quantum simulation with no hardware or gate-noise simulation.
