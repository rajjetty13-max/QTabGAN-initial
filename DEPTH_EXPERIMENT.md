# Improving the readout circuit: test depth without changing the classical model

This experiment varies the number of quantum variational layers while retaining the eight-qubit paper RY/RZ blocks, ordered circular CNOTs and experimental final-Hadamard readout. The mapper, D(x,y) discriminator, preprocessing, objective, learning rates and batch size are unchanged.

The paper describes L variational layers without fixing a reproducible value. One, two and three layers therefore test an unspecified architecture choice. Final-H readout remains an explicit extension rather than a gate claimed to appear in the paper.

## Matched experiment

Every depth is trained for 30 epochs on each of seeds 42, 43 and 44. Validation metrics are measured at epochs 1, 10, 20 and 30. Checkpoints use highest validation macro F1, with earlier epochs winning ties. Final test scores and fixed-epoch-30 results are both reported.

The initial comparison was two versus three layers. A one-layer candidate was added after early validation showed no improvement from three layers. This is adaptive exploratory screening, not a preregistered confirmation.

Changing layer count changes how many random quantum angles are initialized. Without separate random seeds, this would also change the mapper and discriminator's initial weights. The optional `classical_init_seed` now controls their initialization independently. Within each paired seed, all depths receive identical classical weights, identical initial quantum-angle prefixes, identical training RNG and identical split rows. The classical seed is `42000 + seed`; training seed remains `20000 + seed`.

The default for `classical_init_seed` is None, preserving historical initialization and checkpoint compatibility. These new controls are initialized differently from earlier result tables, so compare depths within this experiment rather than using old tables as matched controls.

## Circuit cost

With eight qubits, L layers use 16L angles and 8L CNOTs. The probability vector remains 256-dimensional and the mapper's input size stays fixed. More angles do not automatically mean more useful variation; logical depth, gradients, held-out utility and collapse must be assessed together. These are equal-epoch runs, not equal-compute runs.

The reporting tool plots per-angle RMS gradients rather than comparing raw total gradient norms from differently sized parameter tensors. It also preserves within-class diversity diagnostics. Poor synthetic diversity must not be hidden by a higher accuracy or AUC.

## Run and inspect

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper_readout --condition-discriminator \
  --layers 3 --seed 42 --classical-init-seed 42042 \
  --epochs 30 --eval-every 10 --batch-size 64 \
  --output results/readout_depth_next
```

Use a fresh output folder. Source folders for this experiment are grouped under `results/readout_depth_study/runs/layers_L/seed_S/`. Each contains a source/data manifest, losses, circuit diagram, checkpoint diagnostics, final ROC/PR curves, confusion counts, generated data, fitted preprocessing and best/latest checkpoints.

`scripts/summarize_depth_trials.py` checks paired initialization seeds, training seeds, data hashes, software/source versions and exact split indices before producing comparative plots. Tests confirm independent classical initialization, matching quantum-angle prefixes, gradients through the additional layer and IonQ export correctness.

This is local analytic simulation with backpropagation. The paper's finite-shot training remains a documented difference. Previously examined Loan splits are reused; test metrics are exploratory rather than independent confirmation. No hardware or gate-noise experiment is included.


## Completed results

All nine runs completed. Two layers retained the best mean validation macro F1 (0.5409), compared with 0.5146 for one and 0.5137 for three. One layer is retained as a lower-cost candidate: its fixed-epoch-30 mean test AUROC was **0.7808** versus **0.7383** for two, using **8 versus 16 CNOTs**, but accuracy and macro F1 were lower. Three layers did not justify the extra cost under the declared selection criterion. All depths remained collapsed.

[Full report and gate costs](results/readout_depth_study/analysis/REPORT.md) · [Validation curves](results/readout_depth_study/analysis/validation_curves.png) · [Circuit diagrams and per-seed evidence](results/readout_depth_study/README.md). All 29 tests passed; historical defaults remain unchanged.


## Continued to 100 epochs

One- and two-layer checkpoints were copied and resumed to 100 epochs for all three seeds, preserving the 30-epoch study. The early one-layer endpoint AUROC advantage did not persist. Validation-selected test macro F1 was 0.5536 versus 0.5603, and AUROC 0.7476 versus 0.7726, for one versus two layers. Both remained collapsed; extra epochs did not solve stability. See the [100-epoch comparison and curves](results/readout_100_study/analysis/REPORT.md).
