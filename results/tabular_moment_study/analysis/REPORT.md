# Mixed-type output and conditional-moment experiment

Completed groups: control, typed_output, moment_matched. Seeds: [42, 43, 44]. Each run completed 30 epochs. Paired split arrays, fitted preprocessing/schema, class order, initialization/training seeds, learning rates, circuit, gain and evaluation schedules were checked.

| Variant | Typed output | Moment weight |
| --- | --- | ---: |
| control | False | 0 |
| typed_output | True | 0 |
| moment_matched | True | 10 |

All groups use G/D learning rates 0.0002/0.0002, quantum initialization gain 4, 2 layers of `paper_readout`, and categorical temperature 0.5.

## Decision from validation

The highest mean validation-selected macro F1 is **moment_matched** (0.7121); control is 0.5735. This ranking uses validation scores only. The test results below did not determine the ranking.

Checkpoints use the first maximum validation macro F1 within each run. Best and final checkpoints are both retained. Means ± sample SD describe these seeds, without establishing statistical significance.

## Held-out predictive utility

| Variant | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | best | 0.5735 ± 0.0728 | 0.7697 ± 0.1312 | 0.6013 ± 0.1000 | 0.7923 ± 0.1012 | 0.3563 ± 0.1838 | 0.5486 ± 0.1853 |
| control | last | 0.4743 ± 0.1307 | 0.5457 ± 0.1644 | 0.4668 ± 0.1141 | 0.7929 ± 0.0612 | 0.3190 ± 0.0097 | 0.8333 ± 0.0651 |
| typed_output | best | 0.6136 ± 0.0595 | 0.8327 ± 0.0332 | 0.6164 ± 0.0610 | 0.7621 ± 0.0610 | 0.3392 ± 0.1731 | 0.4444 ± 0.2191 |
| typed_output | last | 0.3988 ± 0.1863 | 0.5173 ± 0.3041 | 0.4024 ± 0.1707 | 0.6175 ± 0.1610 | 0.1387 ± 0.0497 | 0.6424 ± 0.3674 |
| moment_matched | best | 0.7121 ± 0.0388 | 0.8520 ± 0.0183 | 0.6971 ± 0.0068 | 0.8764 ± 0.0136 | 0.4093 ± 0.0563 | 0.7153 ± 0.1104 |
| moment_matched | last | 0.6451 ± 0.0582 | 0.8443 ± 0.0461 | 0.6470 ± 0.0830 | 0.8046 ± 0.1113 | 0.3487 ± 0.1390 | 0.5174 ± 0.2846 |

The training majority-class proportion is 90.4%. Predicting only that class reaches this accuracy without minority discrimination. Minority recall uses the least frequent training class and its matching confusion-matrix row.

Real-trained classifier test reference: accuracy 0.9813 ± 0.0029, f1_macro 0.9413 ± 0.0104, roc_auc 0.9915 ± 0.0045, average_precision 0.9613 ± 0.0143, minority_recall 0.8125 ± 0.0361.

## Generated-data variation

| Variant | Checkpoint | Numeric within-class std ratio | Category coverage |
| --- | --- | ---: | ---: |
| control | best | 0.0126 ± 0.0044 | 0.4080 ± 0.0319 |
| control | last | 0.0082 ± 0.0044 | 0.3899 ± 0.0002 |
| typed_output | best | 0.0035 ± 0.0006 | 0.9864 ± 0.0233 |
| typed_output | last | 0.0042 ± 0.0010 | 0.9606 ± 0.0018 |
| moment_matched | best | 0.5756 ± 0.0493 | 0.9669 ± 0.0014 |
| moment_matched | last | 0.6000 ± 0.0923 | 0.9655 ± 0.0033 |

The numeric ratio averages the smaller class-specific generated/real standard-deviation ratio across raw numeric features; it does not certify every class-feature pair. Coverage averages the fraction of observed training categories generated per categorical feature.

## Individual runs

| Variant | Seed | Checkpoint | Epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 42 | best | 20 | 0.6571 | 0.9190 | 0.7168 | 0.9074 | 0.5509 | 0.3854 |
| control | 42 | last | 30 | 0.6146 | 0.7250 | 0.5892 | 0.8374 | 0.3173 | 0.7812 |
| control | 43 | best | 20 | 0.5402 | 0.6730 | 0.5459 | 0.7172 | 0.1856 | 0.7500 |
| control | 43 | last | 30 | 0.4524 | 0.5100 | 0.4476 | 0.8181 | 0.3103 | 0.9062 |
| control | 44 | best | 10 | 0.5234 | 0.7170 | 0.5412 | 0.7523 | 0.3325 | 0.5104 |
| control | 44 | last | 30 | 0.3559 | 0.4020 | 0.3635 | 0.7230 | 0.3294 | 0.8125 |
| typed_output | 42 | best | 1 | 0.6654 | 0.8690 | 0.6641 | 0.7496 | 0.3426 | 0.4583 |
| typed_output | 42 | last | 30 | 0.1901 | 0.2170 | 0.2144 | 0.4315 | 0.0814 | 0.8333 |
| typed_output | 43 | best | 10 | 0.6269 | 0.8040 | 0.6372 | 0.8284 | 0.5106 | 0.6562 |
| typed_output | 43 | last | 30 | 0.4577 | 0.5100 | 0.4451 | 0.7125 | 0.1702 | 0.8750 |
| typed_output | 44 | best | 30 | 0.5485 | 0.8250 | 0.5477 | 0.7084 | 0.1645 | 0.2188 |
| typed_output | 44 | last | 30 | 0.5485 | 0.8250 | 0.5477 | 0.7084 | 0.1645 | 0.2188 |
| moment_matched | 42 | best | 20 | 0.7204 | 0.8480 | 0.6899 | 0.8801 | 0.3944 | 0.6979 |
| moment_matched | 42 | last | 30 | 0.6870 | 0.8940 | 0.6918 | 0.8480 | 0.3766 | 0.4375 |
| moment_matched | 43 | best | 20 | 0.7460 | 0.8720 | 0.7033 | 0.8613 | 0.3620 | 0.6146 |
| moment_matched | 43 | last | 30 | 0.5786 | 0.8030 | 0.5513 | 0.6782 | 0.1979 | 0.2812 |
| moment_matched | 44 | best | 30 | 0.6698 | 0.8360 | 0.6980 | 0.8877 | 0.4716 | 0.8333 |
| moment_matched | 44 | last | 30 | 0.6698 | 0.8360 | 0.6980 | 0.8877 | 0.4716 | 0.8333 |

[Validation trajectories](validation_curves.png) · [Variation trajectories](diversity_curves.png) · [Loss trajectories](loss_curves.png) · [All per-seed metrics](final_scores.csv) · [Protocol audit](protocol_audit.csv)

## Interpretation and scope

These splits were examined in earlier work; this is exploratory evidence, not independent confirmation. Additional output sampling changes the random-number stream even though paired runs start from matching training seeds. Identical subsequent latent/minibatch draws are not claimed.

Typed output adds a bounded numeric sigmoid and straight-through Gumbel categorical samples. It introduces classical randomness at training and inference. Conditional moment matching adds an objective based only on training rows. Both are explicit extensions to the supplied paper, which specifies a linear mapper output. The retained final-H readout is also an extension; analytic circuit training differs from its finite-shot estimator. These results do not establish a faithful reproduction or a quantum advantage.

Generator losses include the moment penalty when enabled and therefore are not directly comparable across objectives. Additional Gumbel randomness can raise apparent diversity without proving that the quantum input supplies that diversity.

## Source audit

Shared recorded source hashes were checked. The regularization helper may be absent from control manifests because those runs started before that file arrived; all recorded helper hashes must agree.

All shared source hashes match.

The exact launch source is preserved in `results/tabular_moment_study/source`. Its hashes were verified against the manifests and are listed in [source_audit.csv](source_audit.csv). Use this snapshot to reproduce the experiment; the live working tree may contain later fixes.

Source runs:

- `results/tabular_moment_study/runs/control/seed_42`
- `results/tabular_moment_study/runs/control/seed_43`
- `results/tabular_moment_study/runs/control/seed_44`
- `results/tabular_moment_study/runs/typed_output/seed_42`
- `results/tabular_moment_study/runs/typed_output/seed_43`
- `results/tabular_moment_study/runs/typed_output/seed_44`
- `results/tabular_moment_study/runs/moment_matched/seed_42`
- `results/tabular_moment_study/runs/moment_matched/seed_43`
- `results/tabular_moment_study/runs/moment_matched/seed_44`
