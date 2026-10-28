# The team's training and diagnostics workflow

**Status:** The long run below was stopped at epoch 227 to prioritize [circuit design](CIRCUIT_DESIGN.md). It did not reach 1,000 epochs. Its outputs, including the epoch-220 checkpoint, were subsequently removed during cleanup. Treat the command below as the documented training protocol, not a currently running job.

We are testing whether giving QTabGAN more training actually improves its generated data. A changing GAN loss is not enough: the generated rows must also vary, preserve useful relationships, and support predictions on real rows the model has never seen.

## One model implementation

Everyone should work in this repository. The shared implementation is `qtabgan/model.py`, `qtabgan/quantum_generator.py`, `qtabgan/networks.py`, and `qtabgan/preprocessing.py`. There is no separate experimental copy of the quantum model.

Use **`extended_benchmark.py` for new long QTabGAN experiments**. It calls that shared implementation and adds restartable training and validation diagnostics. `benchmark.py` preserves the older QTabGAN/SDV CTGAN comparison protocol and provides the shared evaluation functions. `train.py` and `train_demo.py` are introductory examples, not sources of benchmark numbers. SDV CTGAN is different from the paper's CTAB-GAN and CTAB-GAN+ baselines.

## Run the longer experiment

From the repository root, with the dependencies in README installed:

```bash
export MPLCONFIGDIR=/tmp/qtabgan-mpl
.venv/bin/python -u extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --epochs 1000 --eval-every 100 --batch-size 64 \
  --output results/loan_extended_1000
```

This uses all 5,000 Loan rows: 3,000 for training, 1,000 for validation, and 1,000 for final testing, stratified by the target class. Preprocessing learns only from the training rows. Seed 42 and the row indices are saved. Training uses the existing 8-qubit, 2-layer circuit, analytic probabilities, and existing learning rates. Hardware and hardware noise are separate follow-up work.

To restart after an interruption, repeat the exact command with `--resume`. Checkpoints retain both Adam optimizers and the Python, NumPy and Torch random states. The saved epoch is resumed; at most nine completed epochs may need repeating after a crash. New runs refuse to overwrite an existing output directory. Resume rejects changed data, training source, library versions, batch size, seed, or evaluation interval. Older `model.pt` files contain weights only and cannot recover the original optimizer trajectory.

Set the epoch budget before looking at test results. A completed run can technically be extended, but then its previously viewed test set is no longer a fresh holdout for that new exploratory decision. After a crash, the loss CSV can be ahead of `latest.pt`; resumed training rebuilds the CSV from that checkpoint.

## What happens to the data?

The bank's real data does **not** become amplitudes fed into the quantum generator. The circuit receives independent random angles. Hadamard, RY, RZ and CNOT gates turn those angles and trainable parameters into 256 probabilities. A classical mapper combines those probabilities with the requested class and outputs a generated encoded row.

Real rows are encoded separately: numeric features are clipped at training percentiles and scaled to [0, pi]; categorical features become one-hot vectors. The discriminator compares real and generated encoded rows. The historical default sees features only; `--condition-discriminator` also supplies labels, testing D(x,y) in paper §3.2. See PAPER_CONDITIONING.md for the current matched experiment. Decoding can produce invalid numeric values because mapper outputs are unconstrained; the raw diagnostics report these rather than hiding them behind re-encoding/clipping.

## Where the evidence lives

| File or directory under the run folder | What it tells us |
| --- | --- |
| `manifest.json`, `split.npz`, `preprocessor.joblib` | Data/source hashes, environment, exact split and training-fitted encoding |
| `inputs/input_histograms.png` | Actual training input distributions |
| `inputs/input_correlations.png` and `.csv` | Correlations between original numeric inputs |
| `inputs/encoded_correlations.png` and `.csv` | The 20 encoded columns with highest variance |
| `inputs/input_summary.csv`, `encoded_summary.csv`, `class_counts.csv`, `category_counts.csv` | Numeric ranges, encoded ranges, imbalance and category counts |
| `inputs/data_flow.json`, `training_flow.png` | Actual dimensions and the route through the network |
| `training_circuit.png`, `training_circuit.txt` | Exact training circuit layout at initialization; parameters subsequently train |
| `losses.png`, `losses.csv` | Generator loss, discriminator loss, quantum gradient norm at each epoch |
| `validation_trajectory.png`, `validation_scores.csv` | Whether accuracy, macro F1, AUROC and average precision improve with more training |
| `validation/epoch_*/` | Validation evidence at epoch 1, each evaluation interval and the final epoch |
| `latest.pt`, `best.pt` | Restart state and checkpoint selected by highest validation macro F1 |
| `test/real/`, `test/last/`, `test/best/` | Final real-data reference, final generator, validation-selected generator |
| `final_report.json` | Final test numbers and selected epoch, written only after training finishes |

Every evaluated checkpoint saves `utility.json`, `confusion.png`, `auc_curves.png`, `predictions.csv`, `roc_curve.csv` and `precision_recall_curve.csv` for binary classification. The classifier is a 100-tree random forest trained on generated rows with their requested labels, then evaluated on held-out real rows. The real-data reference trains the same classifier on real training rows. AUC here measures that downstream classifier, not the GAN discriminator.

`feature_diagnostics.csv` and `cdf.png` compare re-encoded columns. `raw_numeric_diagnostics.csv` checks raw standard-deviation ratios, variation within each class, negative values and values outside the training range. `category_coverage.csv` shows whether categorical outputs collapse to a few choices. `generated_correlations.png` and `correlation_error.png` show preserved or lost feature relationships. Undefined correlations from constant columns remain blank; they are not evidence of a match.

## How to judge success

Look at epoch 100 versus epoch 1,000 **within this new run**. Higher held-out AUROC, average precision and macro F1 together with reasonable within-class diversity and improved distribution fidelity support progress. A nonzero quantum gradient establishes that quantum parameters receive updates; it does not establish useful generated data. Discriminator loss near zero and rising generator loss can indicate that the discriminator easily detects generated rows.

Loan is imbalanced: always predicting “no loan” gets 90.4% accuracy. Chance AUROC is 0.5; a useful precision–recall result should be compared with the 9.6% positive prevalence. Confusion counts reveal whether positive cases are being missed. A high accuracy accompanied by poor macro F1 or zero positive recall is a failed result.

The old `results/loan_100`, `results/adult_100` and interrupted `results/loan_extended_1000` outputs were removed during cleanup. Current retained evidence is indexed in [results/README.md](results/README.md). Their documented numbers are historical references. Their splits and checkpoint protocol differ from this experiment. Their CTGAN scores are **not** a matched comparison with the new 1,000-epoch QTabGAN run. Re-run both models with matching data splits and a stated compute budget before making a new model-ranking claim. A single seed is exploratory; repeat the chosen protocol across seeds before reporting robustness. None of these local runs is yet a verified reproduction of the paper.

## Team handoff

Keep each experiment's command, manifest, complete diagnostic folder and interpretation together. Review the shared model before changing architecture. Make one change at a time after this longer-epoch control, and keep the baseline results. GitHub consolidation needs the team to choose the authoritative remote and branch; this local workflow does not establish that organizational decision or push anything to a remote.

Once a convincing benchmark is established, `IONQ_HARDWARE.md` and `qtabgan/ionq.py` cover checkpoint generation through IonQ Quantum Cloud. Local shot sampling currently models measurement randomness; it is not hardware gate-noise simulation.
