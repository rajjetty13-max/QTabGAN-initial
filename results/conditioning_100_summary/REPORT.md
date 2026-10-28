# 100-epoch paper-based training experiment

Original eight-qubit/two-layer circuit, ReLU mapper with linear output, and label-conditioned D(x,y). Three full-Loan runs, seeds 42/43/44; 60/20/20 split per seed; batch size 64. Evaluation at epoch 1 and every 10 epochs. Analytic local simulation; no hardware noise.

The curves show downstream classifier utility on validation rows. They are not discriminator accuracy. Thin lines are individual seeds; the black line is the mean and shading is one sample standard deviation. All measured points are retained without smoothing.

## Validation trajectory

| Epoch | Accuracy | Macro F1 | AUROC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.6783 | 0.4560 | 0.5037 | 0.0995 |
| 10 | 0.7337 | 0.5606 | 0.6987 | 0.2778 |
| 30 | 0.5250 | 0.4328 | 0.6908 | 0.2537 |
| 50 | 0.4770 | 0.3967 | 0.5509 | 0.1860 |
| 100 | 0.5573 | 0.4330 | 0.6074 | 0.1788 |

## Final test results

Checkpoints were selected by validation macro F1. “last” is epoch 100; “real” is the real-trained classifier reference. Means ± sample SD across three seeds.

| Checkpoint | Accuracy | Macro F1 | AUROC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| best | 0.7470 ± 0.1509 | 0.5662 ± 0.0884 | 0.7322 ± 0.0592 | 0.2593 ± 0.0740 |
| last | 0.5683 ± 0.1873 | 0.4365 ± 0.1110 | 0.5986 ± 0.1885 | 0.1715 ± 0.0800 |
| real | 0.9813 ± 0.0029 | 0.9413 ± 0.0104 | 0.9915 ± 0.0045 | 0.9613 ± 0.0143 |

Selected epochs: seed 42: 10, seed 43: 100, seed 44: 10.

## Plots and numbers

[Accuracy curve](accuracy_curve.png) · [All validation metrics](validation_learning_curves.png) · [Loss curves](loss_curves.png) · [Diversity curves](diversity_curves.png)

[Per-seed final test numbers](final_test.csv) · [Validation CSV](validation_scores.csv) · [Diversity CSV](diversity.csv)

These exploratory runs reuse previously examined splits. The test results are not a new independent confirmation. Three seeds mix split and initialization variation; they do not establish statistical significance or a verified paper reproduction. No circuit variant was trained in this experiment, so the curves alone cannot attribute a failure to the quantum circuit.

Source runs:
- `results/conditioning_trial_100_seed_42`
- `results/conditioning_trial_100_seed_43`
- `results/conditioning_trial_100_seed_44`


## Interpretation and next step

Extending training to 100 epochs did not produce sustained improvement. Mean validation accuracy was 73.4% at epoch 10 and 55.7% at epoch 100; the majority-class reference is 90.4%. Mean final-epoch test accuracy was 56.8%, compared with 74.7% for validation-selected checkpoints and 98.1% for the real-trained reference. Accuracy is downstream utility, not discriminator accuracy.

At epoch 100, category coverage averaged about 38.9%. The mean minimum within-class numeric standard-deviation ratio averaged approximately 0.245% of real variation. This averages the smaller class-specific ratio per numeric feature; it is not a statement that every class-feature ratio has that value. Diversity remains severely collapsed.

Quantum outputs still respond to latent noise, while fixed-label mapper output variation is very small. Per-seed `mapper_last_diagnostic.json` files record that check. These use different encoded coordinate spaces, so the numerical magnitudes do not establish an information-retention percentage or identify a single cause.

The evidence supports investigating the mapper/training dynamics before attributing the failure to circuit expressivity. Keep the original circuit and the paper's ReLU/linear mapper while testing unspecified optimization settings in controlled validation experiments. The earlier reuploading-circuit test did not improve utility. A new circuit design needs a matched ablation; this 100-epoch experiment alone cannot establish that a circuit redesign is necessary.

All 25 unit tests passed. The first 30 loss epochs exactly match the previous conditioned run for each seed.


## Matched CTGAN reference

A separate SDV CTGAN run completed 100 epochs on the exact seed-42 training, validation and test rows. Its test accuracy is **89.9%**, macro F1 **0.6245**, AUROC **0.8711** and average precision **0.4091**, compared with seed-42 QTabGAN's epoch-100 **47.4%**, **0.3652**, **0.4103** and **0.0866**. This is a paired single-seed comparison, not CTGAN versus the three-seed QTabGAN mean.

CTGAN is stronger here, but its accuracy remains below the 90.4% majority-class reference, and it finds only 22 of 96 positive test cases. It has not solved the benchmark. SDV CTGAN is also distinct from the paper's CTAB-GAN and CTAB-GAN+ baselines.

[Matched comparison and ROC/PR plots](ctgan_comparison/REPORT.md). Source model and diagnostics: `results/ctgan_matched_100_seed_42/`. The reusable runner is `scripts/matched_ctgan.py`; the original CTGAN path remains in `benchmark.py`, and `requirements-lock.txt` pins the external ctgan package.
