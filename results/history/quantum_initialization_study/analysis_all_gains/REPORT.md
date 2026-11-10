# Paired initialization experiment

Factor: `quantum_init_gain`. All other recorded model/training settings are held fixed; paired splits and initialization/training seeds were checked. 30 epochs per run.

Means ± sample SD. Best checkpoints use validation macro F1; last is the shared final epoch. These are exploratory reused-split results, not independent confirmation.

| Factor | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | AP | Within-class std ratio | Raw numeric KS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | best | 0.5409 ± 0.0230 | 0.6937 ± 0.0369 | 0.5453 ± 0.0247 | 0.7428 ± 0.0026 | 0.3701 ± 0.0310 | 0.0026 ± 0.0010 | 0.6137 ± 0.0088 |
| 1 | last | 0.5028 ± 0.0540 | 0.6327 ± 0.0704 | 0.5057 ± 0.0440 | 0.7383 ± 0.0085 | 0.3571 ± 0.0351 | 0.0020 ± 0.0001 | 0.6194 ± 0.0151 |
| 4 | best | 0.5735 ± 0.0728 | 0.7697 ± 0.1312 | 0.6013 ± 0.1000 | 0.7923 ± 0.1012 | 0.3563 ± 0.1838 | 0.0126 ± 0.0044 | 0.5975 ± 0.0123 |
| 4 | last | 0.4743 ± 0.1307 | 0.5457 ± 0.1644 | 0.4668 ± 0.1141 | 0.7929 ± 0.0612 | 0.3190 ± 0.0097 | 0.0082 ± 0.0044 | 0.5876 ± 0.0074 |
| 16 | best | 0.5444 ± 0.0804 | 0.8433 ± 0.0517 | 0.5054 ± 0.0453 | 0.5176 ± 0.1485 | 0.1150 ± 0.0453 | 0.0202 ± 0.0138 | 0.7112 ± 0.2226 |
| 16 | last | 0.4076 ± 0.2646 | 0.6173 ± 0.4177 | 0.3897 ± 0.2226 | 0.5157 ± 0.1482 | 0.1207 ± 0.0442 | 0.0267 ± 0.0069 | 0.5853 ± 0.0054 |

[Validation curves](validation_curves.png) · [Loss curves](loss_curves.png) · [Diversity/distribution checks](diversity_checks.png) · [Per-seed scores](final_scores.csv)

The variation ratio averages the smaller class-specific standard-deviation ratio per raw numeric feature. It is not a guarantee about every class-feature pair. KS compares original numeric distributions, not a sparse one-hot average. No model is declared successful from accuracy alone.

The quantum circuit, probability input, mapper architecture and adversarial objective are unchanged between variants. Gain changes only initialization and is folded into the first Linear layer coefficients. It does not add an inference-time normalization or replace probabilities by non-probability inputs. This initialization is an implementation experiment, not an author-confirmed setting.

The current final-H readout is still an explicit circuit extension; analytic training remains different from the paper’s sampled estimator. No hardware/noise simulation occurred. Source runs:
- `results/quantum_initialization_study/runs/gain_1/seed_42`
- `results/quantum_initialization_study/runs/gain_1/seed_43`
- `results/quantum_initialization_study/runs/gain_1/seed_44`
- `results/quantum_initialization_study/runs/gain_16/seed_42`
- `results/quantum_initialization_study/runs/gain_16/seed_43`
- `results/quantum_initialization_study/runs/gain_16/seed_44`
- `results/quantum_initialization_study/runs/gain_4/seed_42`
- `results/quantum_initialization_study/runs/gain_4/seed_43`
- `results/quantum_initialization_study/runs/gain_4/seed_44`


## Interpretation and candidate decision

**Gain 4 is a promising opt-in candidate.** Mean validation macro F1 increased **0.5409 → 0.5736**. With checkpoints selected using that rule, test accuracy increased **69.4% → 77.0%**, macro F1 **0.5453 → 0.6013**, and AUROC **0.7428 → 0.7923**. Test macro F1 improved on each paired seed, with the largest benefit at seed 42. Results vary substantially across seeds; these reused-split means are not independent confirmation or statistical significance.

The mean minimum within-class numerical standard-deviation ratio increased **0.257% → 1.259%** of real variation, approximately **4.9×**. Raw numerical KS improved **0.6137 → 0.5975** (lower is better). Average category coverage increased only modestly, **39.0% → 40.8%**. These are partial improvements: generated data is still severely collapsed.

Average precision decreased **0.3701 → 0.3563** and its across-seed variation increased. The fixed final-epoch-30 macro F1 was worse (**0.4668 versus 0.5057**), despite higher AUROC. The candidate is not an across-the-board or stable training improvement. It requires a longer matched check before promotion; defaults remain gain 1 and original rates.

Gain 16 increased spread but harmed utility and selected-checkpoint distribution fidelity. It is rejected as a default, along with both learning-rate candidates from the preceding study. Higher accuracy alone was not used to select an intervention: the faster-G majority-class collapse is retained as counterevidence.

The post-training audit found selected-model fixed-label encoded output RMS standard deviation **0.00043 → 0.00231** for gain 1 versus 4, supporting the intended increase in variation through the mapper. These coordinates differ from raw numeric features and quantum probabilities, so that is not an information-retention percentage. Audits are in `../audits/gain_1/` and `../audits/gain_4/`.

All 32 tests passed. The 17-cell teaching notebook reran with exactly unchanged default results. Gain-1 controls exactly reproduce previous baseline loss histories. Actual saved optimizer rates and initialization gains were checked against their manifests. No remote hardware job was run. A conditional checkpoint width-inference bug was also fixed and tested locally.
