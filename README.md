# QTabGAN reproduction workbench

Compare the hybrid quantum generator with the **SDV CTGAN implementation**, using disjoint train/test rows and explicit training labels. This is a research implementation of [QTabGAN, arXiv:2602.12704v2](https://arxiv.org/abs/2602.12704v2), not a verified reproduction. The paper also includes CTAB-GAN and CTAB-GAN+; CTGAN is a different baseline.

## Current team workflow

**Latest continuation:** [One/two-layer readout circuits through 100 epochs](results/readout_100_study/analysis/REPORT.md) show that the early ranking gain did not persist. Best validation-selected checkpoints are retained; both depths still collapse. The 30-epoch studies remain intact.

**Depth study:** [One/two/three-layer circuit results](results/readout_depth_study/analysis/REPORT.md) retain two layers for validation macro F1. One layer halves the CNOT count and improves epoch-30 AUROC, with lower accuracy/F1. [Protocol and reproduction](DEPTH_EXPERIMENT.md) explain the matched initialization.

**Circuit-only improvement:** [Final-H readout](READOUT_CIRCUIT.md) retains the paper blocks and makes all 32 parameters affect probabilities. [Three-seed training results](results/circuit_readout_screen/REPORT.md) show better mean epoch-30 utility but a tradeoff at validation-selected checkpoints; collapse remains. Enable explicitly with `--ansatz paper_readout`.

**CTGAN comparison restored:** [Matched 100-epoch results](results/conditioning_100_summary/ctgan_comparison/REPORT.md) use the same seed-42 split as QTabGAN. `scripts/matched_ctgan.py` runs SDV CTGAN (dependency in `requirements-lock.txt`); `benchmark.py` retains the original CTGAN comparison path. This is a different baseline from the paper’s CTAB-GAN/CTAB-GAN+.

**100-epoch follow-up:** [Learning curves and results](results/conditioning_100_summary/REPORT.md) show unstable accuracy and continued collapse across three seeds. The original circuit and paper-supported D(x,y) were retained.

**Retained results:** [results/README.md](results/README.md) indexes the retained epoch 1–30 comparisons and new epoch 1–100 runs. Earlier 100-epoch outputs, smoke tests and the interrupted long run were removed. Commands and numeric discussions for those older experiments below are historical; reproducing them requires a fresh run.

[Paper conditioning audit](PAPER_CONDITIONING.md): the latest experiment tests the explicit D(x,y) formulation in §3.2, using the original circuit and an optional label-conditioned discriminator (`--condition-discriminator`). The historical feature-only default remains checkpoint compatible. The [completed three-seed conditioning test](results/conditioning_comparison/REPORT.md) improved validation-selected test AUROC and macro F1, but training remained unstable and both variants collapsed.

**Circuit experiment history:** The extended training run was stopped at epoch 227. [CIRCUIT_DESIGN.md](CIRCUIT_DESIGN.md) explains the experimental circuit, its diagrams, and measured checks: 32 effective parameters instead of 24, and logical depth 11 instead of 22. The [matched 30-epoch training test](results/circuit_trial_30_comparison/REPORT.md) found no predictive-utility improvement, and both circuits still showed severe collapse.

Use [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md) for the extended Loan experiment, restart commands, input/correlation plots, and a map from reported numbers to saved evidence. New long QTabGAN runs use `extended_benchmark.py` and the shared `qtabgan/` model; the commands below retain the historical comparison protocol. The new runner selects checkpoints on validation data and reserves a final test set.

## Install and verify

Python 3.12 was used for the recorded runs. For a CPU installation:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
export MPLCONFIGDIR=/tmp/qtabgan-mpl
python -m unittest discover -s tests -v
python benchmark.py --epochs 2 --qubits 4 --max-rows 200 --output results/smoke
```

On Debian, install `python3-venv` if `venv` reports that `ensurepip` is unavailable. The lock file records the installed environment, including CPU Torch. If CPU wheels are not found, add the CPU index above to the lock-file installation.

The bundled breast-cancer dataset is an offline smoke test, **not a dataset in the paper**.

## Full-data classification experiments

Prepare public Adult and Loan files:

```bash
mkdir -p data
curl -L --fail https://archive.ics.uci.edu/static/public/2/adult.zip -o data/adult.zip
curl -L --fail https://raw.githubusercontent.com/gchoi/Dataset/master/UniversalBank.csv -o data/loan.csv
python scripts/prepare_data.py --adult-zip data/adult.zip --loan-csv data/loan.csv
```

Data provenance: Adult comes from [UCI](https://archive.ics.uci.edu/dataset/2/adult); the Loan CSV is a [public Universal Bank mirror](https://github.com/gchoi/Dataset/blob/master/UniversalBank.csv). The paper does not specify exact source files or row hashes. We combine both UCI Adult files, normalize trailing label periods, keep `?` as a category, and retain 48,842 rows rather than inventing the paper's approximately 48,000-row selection. Loan has 5,000 rows; its ID column is removed, ZIP code and discrete attributes are categorical. These decisions are explicit reproduction assumptions.

```bash
python -u benchmark.py --data data/adult.csv --target income \
  --dataset-name adult --categorical workclass education marital-status \
  occupation relationship race sex native-country --epochs 100 \
  --batch-size 256 --checkpoints 30 --output results/adult_100

python -u benchmark.py --data data/loan_prepared.csv --target PersonalLoan \
  --dataset-name loan --categorical 'ZIP Code' Education SecuritiesAccount \
  CDAccount Online CreditCard --test-size .3 --epochs 100 --checkpoints 30 \
  --output results/loan_100

python scripts/compare_paper.py results/adult_100 results/loan_100
python scripts/diagnose_tables.py results/adult_100 results/loan_100
python scripts/plot_comparison.py results/adult_100 results/loan_100
```

Use `--seeds 42 43 44` for variation across splits and initialization. A single seed is exploratory evidence. CPU quantum simulation can take tens of minutes for Adult. `--max-rows` is available for debugging but invalidates full-data comparisons. Use a new output directory for each experiment; existing files can be overwritten.

## Outputs and interpretation

Each run saves:

- `manifest.json`: complete arguments, library versions, dataset SHA256, label order, and reproduction status. New runs also record Python source hashes; the initial long runs predate that addition.
- `seed_*/split.npz`: exact input row positions used for training and testing.
- `seed_*/preprocessor.joblib`: fitted transformation for new runs. Initial runs made before this artifact was introduced can reconstruct it from the saved split and original CSV.
- `seed_*/real/utility.json`: train-on-real/test-on-real reference.
- `seed_*/*/metrics.json`: train-on-synthetic/test-on-real accuracy, F1, AUROC, average precision, confusion counts, utility gaps, and distribution diagnostics.
- `losses.csv`, `losses.png`: GAN losses and QTabGAN quantum gradient norms.
- `auc_curves.png`, `confusion.png`, `cdf.png`, `feature_diagnostics.csv`: predictive and distribution checks.
- `epoch_30/`, `epoch_100/`: QTabGAN evaluation snapshots. Adam moments persist between stages. Evaluation consumes random numbers, so checkpoint schedules are part of the recorded protocol.
- `synthetic.csv`, model weights, and QTabGAN circuit diagrams/resources.
- `summary.csv`, `aggregate.csv`: individual runs and means/sample standard deviations. One seed has an undefined standard deviation, not an uncertainty estimate.

Both downstream models use a 100-tree RandomForest with the same seed and are tested on the same unseen real rows. Transformations are fitted only on training data. QTabGAN generates training-sized data with training label frequencies; CTGAN generates the joint table including labels. CTGAN uses its own preprocessing, WGAN-GP objective, `pac=2`, and the stated classical widths. Their losses are not directly comparable. Equal epochs also do not imply equal runtime, parameters, or circuit evaluations. Recorded `seconds` include generation, plotting and downstream evaluation (including QTabGAN checkpoints), so they are not pure training timings.

Distribution metrics use decoded/re-encoded features: categorical mapper outputs are decoded by argmax before evaluation. The numerical diagnostics are in clipped [0, pi] coordinates. New runs include `unique_pair_correlation_difference`, which uses the paper's unique-pair convention; `mean_absolute_correlation_difference` is the legacy whole-matrix diagnostic and includes the diagonal. Original-column diagnostics also report numerical unique-pair correlation error. Constant synthetic columns are counted separately; correlation errors omit columns constant in either table and are undefined if fewer than two varying columns remain. Wasserstein/KS scores on one-hot columns are diagnostic summaries, not the paper's categorical JSD. `diagnose_tables.py` adds original-column CDFs, numerical KS/std ratios, out-of-range rates, categorical JSD/coverage, class frequencies, majority-class reference accuracy and per-class numerical diversity. Use these when high-cardinality one-hot columns dominate the averages. JSD is calculated separately on original categorical values, using natural logarithms.

F1 macro, weighted, and binary are reported because the paper does not identify its averaging convention. AUROC/AP curves currently cover binary tasks; multiclass tasks retain confusion matrices and macro/weighted F1. Regression utility helpers exist, but the benchmark CLI intentionally handles classification; regression reproduction remains future work.

The paper's Table 3 reports (the provided PDF hash and reference values are archived in `results/paper_reference.json`):

| Dataset | Real accuracy | Synthetic accuracy | Accuracy gap | Real F1 | Synthetic F1 | F1 gap |
|---|---:|---:|---:|---:|---:|---:|
| Adult (best) | .7760 | .7740 | .0020 | .7619 | .7539 | .0080 |
| Loan (worst) | .9600 | .9380 | .0220 | .9570 | .9291 | .0279 |

The Table 3 discussion identifies Adult and Loan. A small utility gap alone is insufficient: inspect absolute utility, class-specific counts, majority-class accuracy, feature diversity and fidelity. Checkpoints inspect the test set for diagnosis; use a separate validation split before tuning or selecting a final model.

## Architecture and assumptions

`qtabgan/quantum_generator.py` is the sole definition of the quantum gates, shared by training, finite-shot sampling, drawing and resource reports:

```text
uniform random angles in [0, pi]
  -> H + RY(noise) on each wire
  -> L repetitions of RY(theta), RZ(theta), ordered circular CNOT chain
  -> 2**n basis probabilities
  -> concatenate one-hot label
  -> ReLU mapper (128, 128), linear feature output
  -> ReLU discriminator (128, 64, 32), one real/fake logit
```

The paper uses 8 qubits, so that is the default. Trainable angles start uniformly in [0, 2*pi). The source does not specify all hidden widths, layer counts, optimizer settings, epochs, exact noise encoding, or downstream model settings; these are recorded implementation choices. Its prose is ambiguous about repeating Hadamards; the recorded runs apply them once. `--hadamards-per-layer` explicitly tests the literal repeated-H interpretation (the same initial noise encoding, plus H gates before later variational layers). This is a controlled architecture option; the recorded 100-epoch runs do not use it. The historical default discriminator sees features only, following Figure 4. Section 3.2 also explicitly defines D(x,y); `--condition-discriminator` tests that equation-based interpretation by concatenating class labels at the discriminator input. This wiring is an explicit assumption; see PAPER_CONDITIONING.md.

Training defaults to batched analytic probabilities and simulator backpropagation, which accelerates the same ideal circuit. `--diff-method parameter-shift` retains a slower circuit-by-circuit alternative; a test verifies agreement between both gradients. The paper describes finite-shot probability estimation and parameter-shift, so analytic training is an explicit deviation. `--finite-shot-training --diff-method parameter-shift --shots 1024` enables finite-shot probability estimates and parameter-shift during training; this path is tested but was not used for the long runs. It is much slower. `--finite-shot-evaluation --shots 1024` uses empirical shot histograms for final generation. Default evaluation is analytic with the default analytic-training configuration. With `--finite-shot-training`, the training QNode already uses sampled probabilities even without `--finite-shot-evaluation`. Neither option simulates gate noise. For a paired comparison from a completed, trusted local checkpoint, run `python scripts/evaluate_shots.py results/loan_100 --shots 1024`; both modes use the same latent angles and training labels, and outputs are written under `qtabgan/paired_1024_shots/`. This isolates generation-time shot uncertainty; training remains analytic. Numerical jitter is disabled in benchmark preprocessing to keep evaluation deterministic; the general preprocessor retains its original `noise_std=1e-5` option. No hardware or quantum advantage is claimed.

The architecture audit for the recorded long runs is:

| Component | Paper | Recorded implementation |
|---|---|---|
| Quantum width | 8 qubits / 256 probabilities | Matches |
| Trainable gates and entanglement | RY, RZ, directed circular CNOTs | Matches |
| Label conditioning | One-hot label appended to mapper input | Matches |
| Mapper/discriminator activations | ReLU hidden layers; linear mapper output | Matches; discriminator uses logits with BCE |
| Circuit depth and hidden widths | Not numerically specified | 2 layers; mapper (128,128), discriminator (128,64,32) |
| Hadamard placement | Ambiguous layer wording | Once initially; repeated-H option available |
| Noise encoding | Encoding operation not specified | RY angles sampled uniformly in [0,pi] |
| Training probabilities/gradients | Finite-shot estimation; parameter-shift described | Analytic probabilities/backprop; finite-shot parameter-shift option tested separately |
| Numerical preprocessing jitter | Small Gaussian jitter | Disabled for recorded benchmarks; general preprocessor supports it |

The legacy `train.py` now defaults to 8 qubits and one-hot encodes numeric classification labels. `--task regression` retains scalar conditioning for exploratory use; it is not a validated regression benchmark.

`qtabgan/model.py` trains with alternating BCE discriminator and non-saturating generator updates. `qtabgan/preprocessing.py` clips training percentiles and scales numerical features, encodes categories, and decodes generated tables. `qtabgan/evaluation.py` requires separate real-training and synthetic-training labels. The original implementation incorrectly borrowed training labels from the test-label array; that has been removed.

## Circuit simplification and the next noise phase

```bash
PYTHONPATH=. python scripts/circuit_report.py --qubits 8 --layers 2
PYTHONPATH=. python scripts/circuit_report.py \
  --checkpoint results/loan_100/seed_42/qtabgan/model.pt \
  --output results/loan_100/seed_42/qtabgan/optimized_circuit
```

This writes original and simplified exact circuit diagrams and logical resources. It merges adjacent RY rotations and removes the last layer's RZ gates: those phases do not change probabilities before the final computational-basis permutation. Eight random test inputs match to numerical precision. This preserves measured probabilities, not necessarily the statevector. The report gives 64 -> 48 gates and logical depth 22 -> 20, while retaining all 16 CNOTs. The simplified circuit retains 24 of the original 32 trainable angles; final-layer RZ angles have no effect on ideal output probabilities. It does not claim a reduction in backend-native entangling depth or apply the optimized circuit to benchmark training.

After the noiseless baseline is satisfactory, use the integration pointers in [qtabgan/README_CODE_NOTES.md](qtabgan/README_CODE_NOTES.md) for IonQ and IBM simulation. Hardware noise sweeps and hardware execution have not been run.

## Findings from the initial runs

The 100-epoch Loan QTabGAN run (seed 42) does **not** reproduce Table 3. Its real-trained accuracy is 0.9827 and synthetic-trained accuracy is 0.5780, an absolute gap of 0.4047 versus the paper's 0.0220. Macro F1 is 0.4427, binary F1 0.1682, AUROC 0.5877 and average precision 0.1578. Confusion counts are `[[803, 553], [80, 64]]` (rows=true, columns=predicted, classes 0/1). Majority-class accuracy is 0.9040. These are failing utility results even though quantum gradients are finite and nonzero.

At epoch 30, Loan synthetic-trained accuracy was 0.5400 and AUROC 0.7285. At epoch 100, accuracy rises but AUROC and macro F1 worsen. Extra epochs alone do not resolve this failure. Original-column diagnostics show mean numerical KS 0.5729, mean categorical JSD 0.2324 and average category coverage 0.3892. One-hot-averaged KS (0.0181) is misleading here because hundreds of ZIP-code indicator columns dominate it.

These runs differ from the paper in documented preprocessing and unspecified evaluator/training settings. In particular, Loan drops the ID column, leaving 12 predictors versus Table 1's nominal 13. A future protocol sensitivity check can retain ID and vary ZIP-code treatment, but should not select preprocessing by test-set performance. The current unconditional discriminator also cannot enforce label consistency directly. Further work should distinguish this structural limitation from preprocessing and optimization problems before attributing effects to quantum hardware noise.

The matching 100-epoch Loan CTGAN run reaches accuracy 0.9147, macro F1 0.7446, AUROC 0.8915 and average precision 0.5473. Its accuracy gap is 0.0680. CTGAN performs substantially better than this QTabGAN implementation on this split, but remains below the real-trained reference. This is a comparison between these configurations, not evidence about quantum advantage.

A paired Loan generation check uses identical latent angles with analytic probabilities and with 1,024 shots. Both give accuracy 0.5787; AUROC is 0.6051 vs 0.5939. Generation-time sampling alone does not resolve collapse. This does not test finite-shot *training*.

The 100-epoch Adult QTabGAN run also fails the reported benchmark: real-trained accuracy is 0.8571, synthetic-trained accuracy 0.6421, and the gap 0.2150 versus the paper's 0.0020. Macro F1 is 0.4639 and AUROC 0.4962. Majority-class accuracy is 0.7607. Original-column numerical KS is 0.8590, categorical JSD 0.4926 and average category coverage 0.1809. This is severe distributional collapse, not a successful reproduction.

`paper_comparison.csv` compares QTabGAN with Table 3 only. CTGAN appears in each run's `summary.csv` and the combined comparison plot; Table 3 does not report a CTGAN target.

Completed 100-epoch accuracy comparison (seed 42, full prepared datasets):

| Dataset | Real-trained RF | CTGAN-trained RF | QTabGAN-trained RF |
|---|---:|---:|---:|
| Adult | 0.8571 | 0.8184 | 0.6421 |
| Loan | 0.9827 | 0.9147 | 0.5780 |

Adult CTGAN AUROC is 0.8650, macro F1 is 0.7368, and its accuracy gap is 0.0387. The paper's Adult QTabGAN accuracy gap is 0.0020; our QTabGAN gap is 0.2150. On Loan those gaps are 0.0220 versus 0.4047. The paper's benchmark is not reproduced by these runs.

The raw artifacts and plots for those historical 100-epoch experiments were removed during cleanup. See the [retained results index](results/README.md) for current evidence. Both models completed successfully as computations; QTabGAN fails the statistical utility and diversity checks. Nine unit tests and analytic, finite-shot generation, repeated-H, and finite-shot-training smoke runs passed. Hardware noise experiments remain pending; first resolve the noiseless reproduction and the documented architecture/training ambiguities.

## IonQ checkpoint generation

Direct IonQ Quantum Cloud checkpoint inference is implemented in `qtabgan/ionq.py` and `scripts/ionq_generate.py`. It exports the existing circuit, journals job submissions, collects probabilities in the mapper's ordering, and writes synthetic CSV records using the frozen classical mapper. Preparation and local validation are offline; submission is a separate explicit command. See [IONQ_HARDWARE.md](IONQ_HARDWARE.md) for account setup and complete commands. No live IonQ job has been submitted or validated here. The earlier notes about pending hardware experiments still apply to measured hardware results and noise studies.
