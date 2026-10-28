# Results to keep

The retained experiments include the original **1–30 epoch** comparisons and new **1–100 epoch** paper-based conditioning runs. The new runs evaluate at epoch 1 and every 10 epochs. Every training folder retains its losses, circuit diagram, validation checkpoints, final test diagnostics, generated tables, reproducibility manifest, fitted preprocessor and best/latest model checkpoints.

## Start here

**100-epoch readout continuation:** [readout_100_study/](readout_100_study/README.md) preserves six resumed runs and the matched one/two-layer learning curves. The early one-layer AUROC advantage did not persist; validation-selected checkpoints remain better than the endpoints. Original 30-epoch runs are unchanged.

**Latest depth experiment:** [readout_depth_study/](readout_depth_study/README.md) groups nine 30-epoch runs, their circuit diagrams, curves and comparison report. One layer is a lower-cost ranking candidate; two retains the best validation macro F1.

**Latest circuit work:** [Final-H readout experiment](circuit_readout_screen/REPORT.md) reports eight completed 30-epoch runs and a three-seed comparison. [Structural checks and diagrams](readout_circuit_diagnostics/summary.csv) verify 32 effective parameters. `circuit_screen_*_seed_*/` folders contain the source training runs; `readout_circuit_diagnostics/` contains circuit-only evidence.

**CTGAN baseline:** [Matched 100-epoch CTGAN versus QTabGAN, seed 42](conditioning_100_summary/ctgan_comparison/REPORT.md). CTGAN source artifacts live in `ctgan_matched_100_seed_42/`, including `model.pkl`, losses, `validation/`, `test/`, raw distribution diagnostics and generated rows. CTGAN was evaluated only at its final epoch.

**Latest:** [100-epoch report and learning curves](conditioning_100_summary/REPORT.md) · [Accuracy curve](conditioning_100_summary/accuracy_curve.png). The three source folders are `conditioning_trial_100_seed_42/`, `_43/` and `_44/`.

| Report | What it compares |
| --- | --- |
| [Paper conditioning results](conditioning_comparison/REPORT.md) | Main result: original circuit with feature-only versus label-conditioned discriminator, three seeds |
| [Circuit training results](circuit_trial_30_comparison/REPORT.md) | Original versus experimental reuploading circuit, seed 42 |
| [Circuit structure numbers](circuit_comparison/summary.csv) | Gate depth, effective parameters and circuit-only checks |

## Training folders

| Experiment | Control/original | Alternative |
| --- | --- | --- |
| Circuit, seed 42 | `circuit_trial_30_paper/` | `circuit_trial_30_reuploading/` |
| Conditioning, seed 42 | `conditioning_trial_30_control/` | `conditioning_trial_30_conditional/` |
| Conditioning, seed 43 | `conditioning_trial_30_control_seed_43/` | `conditioning_trial_30_conditional_seed_43/` |
| Conditioning, seed 44 | `conditioning_trial_30_control_seed_44/` | `conditioning_trial_30_conditional_seed_44/` |

Within each training folder:

- `validation/epoch_1/`, `epoch_10/`, `epoch_20/`, `epoch_30/`: checkpoint evaluation numbers and plots.
- `losses.csv`, `losses.png`: every training epoch (1–30 in older trials, 1–100 in the latest runs).
- `training_circuit.txt`, `training_circuit.png`: circuit layout.
- `test/real/`, `test/best/`, `test/last/`: final utility, AUC curves and confusion counts.
- `final_report.json`: final results and validation-selected epoch.
- `best.pt`, `latest.pt`, `preprocessor.joblib`, `split.npz`: model and data state. Some binary artifacts are gitignored; retaining them locally does not include them in a Git commit.

`conditioning_comparison_seed_42/`, `_43/` and `_44/` provide the individual paired reports and plots behind the aggregate. `paper_reference.json` preserves the paper's benchmark numbers and PDF hash; it is reference data, not a successful local reproduction.

The earlier 100-epoch Adult/Loan outputs, smoke tests, old circuit simplification outputs, interrupted 1,000-epoch run and their obsolete comparison files were removed during cleanup. Historical numbers mentioned elsewhere in the documentation do not imply those raw artifacts are still present. Source code, datasets, logs and hardware bundles were retained.
