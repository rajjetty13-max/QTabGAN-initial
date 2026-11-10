# Circuit depth comparison

Paper-based readout circuit, 3 paired seeds, 100 epochs. Only quantum layer count differs. Independent classical initialization and training RNG are held fixed within each seed; original quantum angle prefixes also match. Equal epochs do not imply equal gate execution or compute.

The readout Hadamards are an explicit extension to the paper. Variational layers retain its RY/RZ operations and ordered CNOT ring. The paper leaves depth unspecified; extra layers test that choice without adding a classical model or objective.

## Test scores

Mean ± sample SD. “best” uses maximum validation macro F1 over the recorded checkpoints. “last” uses epoch 100. Test scores are reported after training, not used to choose checkpoints.

| Layers | Checkpoint | Accuracy | Macro F1 | AUROC | AP |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | best | 0.6890 ± 0.0130 | 0.5536 ± 0.0050 | 0.7476 ± 0.0376 | 0.2912 ± 0.0920 |
| 1 | last | 0.6703 ± 0.0140 | 0.5024 ± 0.0358 | 0.6038 ± 0.1805 | 0.1581 ± 0.0972 |
| 2 | best | 0.6903 ± 0.0496 | 0.5603 ± 0.0195 | 0.7726 ± 0.0288 | 0.2934 ± 0.0738 |
| 2 | last | 0.4093 ± 0.0850 | 0.3567 ± 0.0598 | 0.5465 ± 0.1306 | 0.1361 ± 0.0502 |

## Circuit cost

| Layers | Parameters | Gates | CNOTs | Logical depth |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 48 | 8 | 13 |
| 2 | 32 | 72 | 16 | 23 |

## Evidence

[Validation curves](validation_curves.png) · [Loss/gradient curves](loss_curves.png) · [Diversity curves](diversity_curves.png) · [Per-seed scores](final_scores.csv) · [Aggregate](aggregate.csv)

The gradient plot uses L2 gradient norm divided by the square root of the number of angles, to avoid confusing a larger parameter count with stronger per-angle gradients. Within-class variation averages the smaller class-specific standard-deviation ratio per numeric feature. Small values still indicate collapse.

This experiment uses a fresh classical initialization protocol, so previous result tables are not its matched controls. Previously examined Loan splits are reused: the test set is not independent confirmation. Three seeds provide exploratory evidence, not statistical significance or verified reproduction. No noise/hardware experiment was performed.

Reproduce with `extended_benchmark.py --ansatz paper_readout --condition-discriminator --layers L --classical-init-seed C`, retaining all other recorded arguments and using a fresh output folder. Source runs:
- `results/readout_100_study/runs/layers_1/seed_42`
- `results/readout_100_study/runs/layers_1/seed_43`
- `results/readout_100_study/runs/layers_1/seed_44`
- `results/readout_100_study/runs/layers_2/seed_42`
- `results/readout_100_study/runs/layers_2/seed_43`
- `results/readout_100_study/runs/layers_2/seed_44`


## Continuation outcome

All six runs resumed copies of their exact 30-epoch optimizer/RNG states and reached epoch 100. The originals remain unchanged in `results/readout_depth_study/`. Archived reports, source checkpoint hashes and all six loss prefixes were verified.

The one-layer circuit's fixed-epoch-30 test AUROC advantage did not persist: its mean AUROC fell from **0.7808 at epoch 30 to 0.6038 at epoch 100**. The two-layer endpoint also deteriorated, from **0.7383 to 0.5465**. At epoch 100, one layer scored **67.0% accuracy / 0.5024 macro F1**, and two scored **40.9% / 0.3567**. A relative endpoint win does not mean either generator is working well; majority-class accuracy is 90.4%.

Validation-selected checkpoints are substantially better: one layer gives **68.9% accuracy, 0.5536 macro F1, 0.7476 AUROC**; two gives **69.0%, 0.5603, 0.7726**. The selected epochs are 30/60/20 for one layer and 20/60/70 for two, across seeds 42/43/44. The two-layer model still has the better mean selection criterion; the one-layer model remains a lower-cost candidate using half the CNOTs. These small-sample means do not establish equivalence or statistical significance.

Both remain severely collapsed. The within-class variation summary for selected models averages about **0.20% / 0.22%** of real variation for one/two layers. It averages the smaller class-specific numeric standard-deviation ratio per feature, not every class-feature pair. Extra epochs have not repaired diversity or stability.

For inference experiments, use the validation-selected `best.pt` within each run; `latest.pt` retains the epoch-100 state for training continuation. Keep two layers as the current quality baseline and one as the efficiency candidate. A further improvement needs a controlled circuit/training experiment; simply extending training is not supported by these curves. The default architecture was not changed.

The test sets had already been viewed at epoch 30 and in previous studies; these scores are exploratory continuation results, not independent confirmation or a reproduced paper benchmark. Structural noise advantages were not measured.
