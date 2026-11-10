# Conditional-moment continuation through 100 epochs

Three exact continuations, seeds 42/43/44, from epoch 30 to total epoch 100. Weights, Adam states and RNG state were resumed; the split, preprocessing, circuit, output sampling, moment weight 10, learning rates and batch size were retained. Training uses the preserved source snapshot so later working-tree fixes do not alter the comparison.

## Predictive utility

Best checkpoints use the first maximum validation macro F1 observed through each budget. The 100-epoch search includes all original checkpoints. Fixed checkpoints are the actual epochs 30 and 100. Test scores do not choose checkpoints. Means ± sample SD across three seeds.

| Budget | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | best | 0.7121 ± 0.0388 | 0.8520 ± 0.0183 | 0.6971 ± 0.0068 | 0.8764 ± 0.0136 | 0.4093 ± 0.0563 | 0.7153 ± 0.1104 |
| 30 | last | 0.6451 ± 0.0582 | 0.8443 ± 0.0461 | 0.6470 ± 0.0830 | 0.8046 ± 0.1113 | 0.3487 ± 0.1390 | 0.5174 ± 0.2846 |
| 100 | best | 0.7527 ± 0.0461 | 0.9097 ± 0.0031 | 0.7538 ± 0.0199 | 0.8932 ± 0.0097 | 0.5229 ± 0.0448 | 0.5972 ± 0.0739 |
| 100 | last | 0.7079 ± 0.0455 | 0.8973 ± 0.0125 | 0.7042 ± 0.0367 | 0.8515 ± 0.0470 | 0.4690 ± 0.0611 | 0.4653 ± 0.0694 |

## Numeric variation and categories

| Budget | Checkpoint | Numeric within-class std ratio | Category coverage |
| ---: | --- | ---: | ---: |
| 30 | best | 0.5756 ± 0.0493 | 0.9669 ± 0.0014 |
| 30 | last | 0.6000 ± 0.0923 | 0.9655 ± 0.0033 |
| 100 | best | 0.7270 ± 0.0636 | 0.9630 ± 0.0042 |
| 100 | last | 0.7191 ± 0.0846 | 0.9607 ± 0.0035 |

The numeric ratio averages the smaller generated/real within-class standard-deviation ratio per numeric feature. Category coverage averages the fraction of observed training categories generated per feature. Neither metric certifies the full distribution.

## Selected checkpoints by seed

| Seed | Budget | Selected epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Minority recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 30 | 20 | 0.7204 | 0.8480 | 0.6899 | 0.8801 | 0.6979 |
| 42 | 100 | 80 | 0.7660 | 0.9090 | 0.7505 | 0.8944 | 0.5833 |
| 43 | 30 | 20 | 0.7460 | 0.8720 | 0.7033 | 0.8613 | 0.6146 |
| 43 | 100 | 90 | 0.7906 | 0.9130 | 0.7751 | 0.9023 | 0.6771 |
| 44 | 30 | 30 | 0.6698 | 0.8360 | 0.6980 | 0.8877 | 0.8333 |
| 44 | 100 | 70 | 0.7015 | 0.9070 | 0.7358 | 0.8829 | 0.5312 |

## What the longer budget changed

Mean validation-selected test accuracy changed 85.2% → 91.0%; macro F1 changed 0.6971 → 0.7538; AUROC changed 0.8764 → 0.8932. These are reports of the frozen validation selections, not a rule to choose an epoch from its test score.

More epochs need not improve utility. Validation-best F1 cannot decrease when earlier checkpoints remain candidates, but test utility and the final checkpoint can worsen. Accuracy should be read alongside minority recall and macro F1; the majority-only accuracy reference is 90.4%.

[Validation curves](validation_curves.png) · [Diversity curves](diversity_curves.png) · [Per-seed scores](final_scores.csv) · [Paired changes](selected_paired_deltas.csv)

## Verification and scope

All 30-epoch loss and validation prefixes match exactly. Split arrays and serialized preprocessors match, retained source checkpoint hashes match continuation provenance, Adam step counters advance from 30 to 100 epochs, and preserved launch-source hashes match every run manifest. See [continuation_audit.csv](continuation_audit.csv) and [source_audit.csv](source_audit.csv).

The 30-epoch experiment and these splits were already examined. This continuation is exploratory, not independent confirmation or a significance test. The mixed-type output and moment objective extend the supplied paper. Categorical generation includes classical Gumbel randomness; these curves do not establish a quantum advantage. No threshold calibration is applied in this report.

Original study: `results/tabular_moment_study`. Continuation study: `results/tabular_moment_100_study`.
