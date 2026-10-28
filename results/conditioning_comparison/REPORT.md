# Paper-based discriminator conditioning: three-seed test

The original quantum circuit and mapper were retained. Only discriminator label conditioning was varied, following D(x,y) in paper §3.2. All six runs completed 30 epochs on full Loan data with paired splits and training seeds.

## Results

Means ± sample standard deviations over seeds 42, 43 and 44. “best” uses highest validation macro F1 from epochs 1/10/20/30, then reports performance on the shared per-seed test set. “last” reports epoch 30. These are exploratory results across three varying splits and initializations, not confidence intervals.

| Checkpoint | Discriminator | Accuracy | Macro F1 | AUROC | Average precision |
| --- | --- | ---: | ---: | ---: | ---: |
| best | paper + D(x) | 0.7763 ± 0.1047 | 0.5184 ± 0.0524 | 0.5767 ± 0.1379 | 0.2035 ± 0.1359 |
| best | paper + D(x,y) | 0.7407 ± 0.1490 | 0.5767 ± 0.0900 | 0.7239 ± 0.0484 | 0.3106 ± 0.1060 |
| last | paper + D(x) | 0.6343 ± 0.0678 | 0.5033 ± 0.0463 | 0.7096 ± 0.0122 | 0.3057 ± 0.0630 |
| last | paper + D(x,y) | 0.5367 ± 0.0978 | 0.4425 ± 0.0600 | 0.7137 ± 0.0220 | 0.2924 ± 0.0655 |

## What this establishes

Validation-selected D(x,y) checkpoints improved macro F1 and AUROC on each of the three paired seeds. Their selected epoch was 10 for all seeds; controls selected epochs 1, 1 and 30. At the common final epoch 30, the conditional variant did not show a consistent utility improvement. This is evidence for an early-checkpoint benefit, not stable training or successful paper reproduction.

Both variants still collapse. Mean category coverage remains around 39–42%, raw numeric KS errors are large, and the mean minimum within-class standard-deviation ratio remains below 1% in the selected models. That summary averages the smaller class-specific ratio for each numeric feature; it does not assert every class-feature ratio is below 1%.

In seed 42, the mapper diagnostics at epoch 30 show quantum probabilities respond to noise, while fixed-label encoded outputs vary only weakly. This is consistent with loss of diversity in the generator/mapper path, not proof of a single causal bug. The diagnostic uses mixed encoded coordinates, so sensitivity magnitudes across spaces cannot be interpreted as a direct information-retention percentage.

The paper specifies ReLU hidden layers and a linear mapper output; those were preserved. No output sigmoid, auxiliary classifier or replacement loss was added. The next controlled paper-based experiment should investigate the unspecified optimization settings and mapper widths, using validation metrics and diversity together. Previously viewed test sets must not be used to select the next settings.

## Evidence

- [Paired score plots](paired_scores.png)
- [All checkpoint numbers](all_results.csv)
- [Aggregate numbers](aggregate.csv)
- [Seed 42: curves and complete metrics](../conditioning_comparison_seed_42/REPORT.md)
- [Seed 43: curves and complete metrics](../conditioning_comparison_seed_43/REPORT.md)
- [Seed 44: curves and complete metrics](../conditioning_comparison_seed_44/REPORT.md)

Each source run retains losses, confusion matrices, ROC/precision–recall curves and their underlying numbers, generated tables, raw distribution diagnostics, manifests, and best/latest checkpoints.

Seed 42 mapper diagnostics: [control](../conditioning_trial_30_control/mapper_last_diagnostic.json), [D(x,y)](../conditioning_trial_30_conditional/mapper_last_diagnostic.json).

Architecture assumption: the paper names D(x,y) but its diagram does not explicitly wire the class input. This experiment concatenates the one-hot class at the discriminator input. Historical feature-only checkpoints remain compatible via an opt-in flag. Analytic simulation/backprop, implementation-chosen hidden widths and dataset/evaluator assumptions remain deviations or unspecified details. No hardware or gate-noise simulation was used. All 25 unit tests passed.
