# Circuit depth comparison

Paper-based readout circuit, 3 paired seeds, 30 epochs. Only quantum layer count differs. Independent classical initialization and training RNG are held fixed within each seed; original quantum angle prefixes also match. Equal epochs do not imply equal gate execution or compute.

The readout Hadamards are an explicit extension to the paper. Variational layers retain its RY/RZ operations and ordered CNOT ring. The paper leaves depth unspecified; extra layers test that choice without adding a classical model or objective.

## Test scores

Mean ± sample SD. “best” uses maximum validation macro F1 over epochs 1/10/20/30. “last” uses epoch 30. Test scores are reported after training, not used to choose checkpoints.

| Layers | Checkpoint | Accuracy | Macro F1 | AUROC | AP |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | best | 0.6853 ± 0.0193 | 0.5201 ± 0.0597 | 0.6687 ± 0.1651 | 0.2611 ± 0.1439 |
| 1 | last | 0.5793 ± 0.1201 | 0.4831 ± 0.0690 | 0.7808 ± 0.0097 | 0.3719 ± 0.0163 |
| 2 | best | 0.6937 ± 0.0369 | 0.5453 ± 0.0247 | 0.7428 ± 0.0026 | 0.3701 ± 0.0310 |
| 2 | last | 0.6327 ± 0.0704 | 0.5057 ± 0.0440 | 0.7383 ± 0.0085 | 0.3571 ± 0.0351 |
| 3 | best | 0.6720 ± 0.0734 | 0.5376 ± 0.0541 | 0.7606 ± 0.0241 | 0.3765 ± 0.0277 |
| 3 | last | 0.6127 ± 0.0261 | 0.4966 ± 0.0178 | 0.7250 ± 0.0110 | 0.3280 ± 0.0165 |

## Circuit cost

| Layers | Parameters | Gates | CNOTs | Logical depth |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 48 | 8 | 13 |
| 2 | 32 | 72 | 16 | 23 |
| 3 | 48 | 96 | 24 | 33 |

## Evidence

[Validation curves](validation_curves.png) · [Loss/gradient curves](loss_curves.png) · [Diversity curves](diversity_curves.png) · [Per-seed scores](final_scores.csv) · [Aggregate](aggregate.csv)

The gradient plot uses L2 gradient norm divided by the square root of the number of angles, to avoid confusing a larger parameter count with stronger per-angle gradients. Within-class variation averages the smaller class-specific standard-deviation ratio per numeric feature. Small values still indicate collapse.

This experiment uses a fresh classical initialization protocol, so previous result tables are not its matched controls. Previously examined Loan splits are reused: the test set is not independent confirmation. Three seeds provide exploratory evidence, not statistical significance or verified reproduction. No noise/hardware experiment was performed.

Reproduce with `extended_benchmark.py --ansatz paper_readout --condition-discriminator --layers L --classical-init-seed C`, retaining all other recorded arguments and using a fresh output folder. Source runs:
- `results/readout_depth_study/runs/layers_1/seed_42`
- `results/readout_depth_study/runs/layers_1/seed_43`
- `results/readout_depth_study/runs/layers_1/seed_44`
- `results/readout_depth_study/runs/layers_2/seed_42`
- `results/readout_depth_study/runs/layers_2/seed_43`
- `results/readout_depth_study/runs/layers_2/seed_44`
- `results/readout_depth_study/runs/layers_3/seed_42`
- `results/readout_depth_study/runs/layers_3/seed_43`
- `results/readout_depth_study/runs/layers_3/seed_44`


## Interpretation and retained candidates

Mean best validation macro F1 was **0.5146 / 0.5409 / 0.5137** for one / two / three layers. Under the declared macro-F1 selection criterion, **two layers remains the baseline**. Three layers increased CNOT count by 50% without improving that criterion; its modest selected-checkpoint test AUROC increase does not justify calling it an overall improvement.

The one-layer candidate offers a different tradeoff. At the shared epoch-30 endpoint it achieved test AUROC **0.7808 ± 0.0097**, versus **0.7383 ± 0.0085** for two layers, while using **8 rather than 16 CNOTs** and logical depth **13 rather than 23**. Its epoch-30 AUROC was higher on each paired seed. However, accuracy decreased **63.3% → 57.9%**, and macro F1 **0.5057 → 0.4831**. Its validation-selected performance was less reliable, including an epoch-1 selection for seed 43. Keep it as a lower-cost ranking candidate, not a replacement selected after looking at test scores.

All three depths still show severe collapse: the mean within-class variation summary remains under 0.3% of real variation in the selected models. This summary averages the smaller class-specific numeric standard-deviation ratio per feature, not every class/feature pair. Neither a higher AUC nor fewer gates establishes a useful synthetic dataset or a reproduced paper benchmark.

Defaults and the prior circuit remain unchanged. Use `--layers 1` for the lower-cost candidate or `--layers 2` for the current F1 baseline, retaining `--ansatz paper_readout --condition-discriminator`. No hardware/noise benefit has been measured. All 29 tests passed.
