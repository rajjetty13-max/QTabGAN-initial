# Paired initialization experiment

Factor: `quantum_init_gain`. All other recorded model/training settings are held fixed; paired splits and initialization/training seeds were checked. 30 epochs per run.

Means ± sample SD. Best checkpoints use validation macro F1; last is the shared final epoch. These are exploratory reused-split results, not independent confirmation.

| Factor | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | AP | Within-class std ratio | Raw numeric KS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | best | 0.5409 ± 0.0230 | 0.6937 ± 0.0369 | 0.5453 ± 0.0247 | 0.7428 ± 0.0026 | 0.3701 ± 0.0310 | 0.0026 ± 0.0010 | 0.6137 ± 0.0088 |
| 1 | last | 0.5028 ± 0.0540 | 0.6327 ± 0.0704 | 0.5057 ± 0.0440 | 0.7383 ± 0.0085 | 0.3571 ± 0.0351 | 0.0020 ± 0.0001 | 0.6194 ± 0.0151 |
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
