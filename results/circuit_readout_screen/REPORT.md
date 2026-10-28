# Circuit-only improvement experiment

All runs use the same original paper variational blocks, eight qubits, two layers, 32 parameters, mapper, D(x,y), optimizer settings and preprocessing. Split rows and training RNG are paired within each seed. The only changes are repeated Hadamards and/or final Hadamards before measurement. Final readout is an explicit extension to the paper, not exact reproduction.

The initial seed-42 screen included four variants. Readout H was taken to seeds 43/44 because it made inactive parameters effective and essentially tied the original validation macro F1 at seed 42. This exploratory selection and reused splits limit any generalization claim. Tests did not select checkpoints: highest validation macro F1 did.

## Per-seed validation-selected checkpoints

| Variant | Seed | Epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 42 | 10 | 0.4775 | 0.5810 | 0.4787 | 0.6696 | 0.1932 |
| Repeated H | 42 | 1 | 0.4735 | 0.8470 | 0.4773 | 0.4246 | 0.0863 |
| Readout H | 42 | 30 | 0.4774 | 0.5730 | 0.4810 | 0.7397 | 0.2735 |
| Repeated H + readout H | 42 | 1 | 0.4683 | 0.7070 | 0.4508 | 0.4224 | 0.0858 |
| Original | 43 | 10 | 0.5776 | 0.7650 | 0.5958 | 0.7624 | 0.3994 |
| Readout H | 43 | 10 | 0.5637 | 0.7160 | 0.5765 | 0.8229 | 0.4205 |
| Original | 44 | 10 | 0.6266 | 0.8760 | 0.6556 | 0.7397 | 0.3393 |
| Readout H | 44 | 10 | 0.5371 | 0.6690 | 0.5457 | 0.7371 | 0.3916 |

## Matched original versus readout

Means ± sample SD for each available seed set; confirm the seed counts below. “last” uses the shared final epoch.

| Variant | Checkpoint | Seeds | Accuracy | Macro F1 | AUROC | AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | best | 3 | 0.7407 ± 0.1490 | 0.5767 ± 0.0900 | 0.7239 ± 0.0484 | 0.3106 ± 0.1060 |
| Original | last | 3 | 0.5367 ± 0.0978 | 0.4425 ± 0.0600 | 0.7137 ± 0.0220 | 0.2924 ± 0.0655 |
| Readout H | best | 3 | 0.6527 ± 0.0729 | 0.5344 ± 0.0487 | 0.7666 ± 0.0488 | 0.3619 ± 0.0779 |
| Readout H | last | 3 | 0.6243 ± 0.0560 | 0.5039 ± 0.0349 | 0.7381 ± 0.0120 | 0.3294 ± 0.0513 |

## Artifacts

[Validation curves](validation_curves.png) · [Paired test scores](paired_test_scores.png) · [All numbers, including diversity](all_results.csv) · [Aggregate](aggregate.csv)

The readout circuit activates all 32 parameter slots versus 24 in the original, adds 8 single-qubit gates, retains 16 CNOTs, and changes logical depth from 22 to 23. Structural evidence is in `results/readout_circuit_diagnostics/`. This is not a noise or hardware advantage.

These are 30-epoch local analytic simulations, not an established paper benchmark. Higher utility is insufficient if diversity remains collapsed. Every source folder includes losses, raw and encoded feature diagnostics, ROC/PR curves, confusion counts, synthetic data, initial circuit diagrams and saved best/latest model states.

Runs:
- `results/circuit_screen_original_seed_42`
- `results/circuit_screen_repeated_h_seed_42`
- `results/circuit_screen_readout_h_seed_42`
- `results/circuit_screen_combined_h_seed_42`
- `results/circuit_screen_original_seed_43`
- `results/circuit_screen_readout_h_seed_43`
- `results/circuit_screen_original_seed_44`
- `results/circuit_screen_readout_h_seed_44`


## Interpretation

At a matched final epoch 30, final-H readout improved the three-seed mean accuracy from **53.7% to 62.4%**, macro F1 from **0.4425 to 0.5039**, AUROC from **0.7137 to 0.7381**, and AP from **0.2924 to 0.3294**. These are exploratory paired means, not a statistically established advantage. The improvement was not uniform across seeds.

When each run instead uses its highest-validation-macro-F1 checkpoint, mean AUROC improved **0.7239 to 0.7666** and AP **0.3106 to 0.3619**, but accuracy fell **74.1% to 65.3%** and macro F1 **0.5767 to 0.5344**. Both checkpoint rules are shown to avoid hiding that tradeoff. A higher ranking score with worse thresholded predictions is not a blanket model improvement.

All variants still show severe distributional collapse: raw numerical KS errors are high and within-class numeric variation remains tiny. Neither this readout change nor the extra active parameters establishes reproduction of the paper's benchmark. The simple readout remains opt-in; the paper baseline stays the default. Repeated layer Hadamards, alone or combined with readout, failed the seed-42 screening criterion and were not promoted.

The change is useful as a circuit-only experiment: it retains the paper's variational blocks, repairs the invisible final phase parameters, and gives a measurable fixed-budget utility gain without altering the classical networks. It adds a logical layer rather than reducing depth. The next longer comparison should predeclare the selection criterion and report both minority-class performance and diversity, not choose settings from the already examined test sets.

All 27 unit tests passed, including readout gate equivalence, gradients, shot probabilities and IonQ export. No remote hardware job was run.
