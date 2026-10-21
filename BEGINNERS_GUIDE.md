# QTabGAN: A Beginner’s Guide to This Project

## What is this project trying to do?

This project tries to create artificial data that resembles a real spreadsheet.

Imagine a table containing people’s ages, incomes, education levels, and income categories. A useful data generator would create new, fictional records that preserve the patterns in the original table. It should produce a variety of realistic records rather than repeatedly outputting almost the same person.

The project compares two generators:

- **CTGAN:** an established generator built using classical neural networks.
- **QTabGAN:** a hybrid generator that combines a simulated quantum circuit with classical neural networks.

The research goal is to reproduce results from the paper *QTabGAN: A Hybrid Quantum-Classical GAN for Tabular Data Synthesis*, then investigate hardware noise before eventually considering quantum hardware.

**The current code runs, but the tested QTabGAN configuration does not reproduce the paper’s results.**

## A few terms before we begin

| Term | Meaning in this project |
|---|---|
| Dataset | A table of records, usually loaded from a CSV file. |
| Row or record | One example, such as one person or one bank customer. |
| Feature | An input column, such as age or education. |
| Target or label | The category we want a classifier to predict, such as an income category. |
| Synthetic data | Artificial records produced by a model. |
| Model | A collection of mathematical operations with adjustable parameters. |
| Training | Repeatedly adjusting those parameters using examples. |
| Epoch | One pass through the training records. |
| Batch | A smaller group of records processed together during training. |
| Neural network | A trainable mathematical function that learns patterns in data. |
| Quantum circuit | A sequence of operations on qubits. Here, those operations run in a simulator. |
| Qubit | The basic unit of quantum information. |
| Seed | A number used to initialize random-number generators so an experiment can be repeated. |
| Checkpoint | A saved model or an evaluation taken at a particular training stage. |

## What is a GAN?

GAN stands for **Generative Adversarial Network**. It has two competing parts.

The **generator** creates artificial records. The **discriminator** examines records and tries to decide whether they came from the real training data or the generator.

Training alternates between improving the discriminator and improving the generator. The generator receives feedback through the discriminator’s predictions.

The intended outcome is that the generator learns the patterns in the real data. This is not guaranteed: GAN training can become unstable or produce too little variety.

## How QTabGAN generates a record

The implemented generator follows this path:

```text
Random numbers
      |
      v
Simulated quantum circuit
      |
      v
Vector of quantum measurement probabilities
      |
      +---- Requested class label
      |
      v
Classical neural network called the mapper
      |
      v
Synthetic feature values
```

### 1. Start with random numbers

The code samples random angles. These provide different inputs so the generator can produce different records.

The exact noise-encoding operation is not specified in the paper. This implementation encodes those angles using RY rotations.

### 2. Run the quantum circuit

The default circuit uses **8 qubits**. It contains:

- Hadamard gates to initialize superposition.
- Trainable RY and RZ rotation gates.
- CNOT gates connecting the qubits in a directed circular pattern.

You do not need to understand the gate mathematics to follow the project. The important point is that the circuit is a trainable function: changing its rotation angles changes the probabilities of its measurement outcomes.

### 3. Obtain a probability vector

Eight qubits have 256 possible computational-basis measurement outcomes, because `2^8 = 256`.

The circuit provides a probability for each outcome. These 256 probabilities add up to one. They are an intermediate representation, not 256 spreadsheet columns or 256 generated records.

### 4. Supply the requested class

For classification tasks, the requested class is represented using **one-hot encoding**.

For example, two income classes can be represented as:

```text
Lower-income class:  [1, 0]
Higher-income class: [0, 1]
```

The mapper receives the probabilities and the requested label together. It is supposed to learn what feature values belong to that class.

The label is an instruction to the generator; it does not guarantee that the generated features actually match the requested class. The evaluation checks whether that relationship was learned.

### 5. Convert the probabilities into a record

A classical neural network, called the **mapper**, converts that combined input into feature values.

The quantum circuit does not directly output a finished spreadsheet row. The mapper performs that conversion. Numerical values are converted back to their original scale, and categorical outputs are decoded into categories.

### 6. Get feedback from the discriminator

The discriminator compares real and generated feature vectors. Its feedback updates the mapper and quantum rotation parameters.

In the current architecture, the discriminator receives the features without the requested label. This follows the paper’s diagram, but it means that the discriminator cannot directly check whether a record matches its supplied class.

## Is this using a real quantum computer?

**No. The experiments described here use local quantum simulation.**

PennyLane constructs and evaluates the quantum circuit, and PyTorch handles the neural networks and training.

There are two ways to obtain the quantum probabilities:

### Analytic probabilities

The simulator calculates the circuit’s probabilities directly. This avoids measurement sampling uncertainty and supports faster simulator backpropagation.

The long, 100-epoch QTabGAN runs used this approach.

### Finite-shot probabilities

A **shot** is one execution and measurement of a circuit. Repeating the circuit produces counts of its outcomes. Dividing each count by the total number of shots estimates its probability.

For example, an outcome observed 256 times in 1,024 shots has an estimated probability of 0.25.

The code supports finite-shot generation and finite-shot training with **parameter-shift gradients**. Parameter-shift estimates how a rotation parameter affects the training objective by evaluating circuits at shifted angles.

These paths have been tested in small runs. The reported 100-epoch training experiments used analytic probabilities and backpropagation.

**Finite-shot sampling is different from hardware noise.** Sampling uncertainty can exist even when the simulated gates are perfect. Hardware noise involves imperfect gates, readout errors, and other device effects. Hardware noise experiments have not been run in this project.

## How the real data is prepared

The benchmark runner first separates the original records into training and test sets.

The training set is used to fit preprocessing and train the generators. The test set is kept out of generator training and used for downstream evaluation.

Numerical preprocessing clips values to training-set percentile limits, normalizes them, and scales them into the interval `[0, pi]`. Categorical features use one-hot encoding.

The benchmark fits these transformations using training rows only. Learning transformation settings from the test set would let information from the test data influence training.

Small numerical jitter described in the paper is disabled in the recorded benchmark runs. The general preprocessor still supports adding it.

## What is CTGAN doing here?

CTGAN provides a classical comparison. The benchmark uses the actual SDV CTGAN package rather than labeling a custom neural network as CTGAN.

Both generators train using the same real training rows, but their internal methods differ:

- QTabGAN generates features for supplied class labels.
- CTGAN learns a joint table containing features and labels, then generates records from it.

The benchmark records these differences. Equal epoch counts do not mean equal computation, equal model size, or identical learning objectives.

CTGAN is also distinct from **CTAB-GAN** and **CTAB-GAN+**, which are separate methods discussed in the paper.

## How do we check whether the generated data is useful?

Looking at a few generated rows is not enough. The evaluation asks whether models trained on those artificial rows can make useful predictions on real records.

It trains three separate random-forest classifiers:

1. A classifier trained on real training records.
2. A classifier trained on CTGAN-generated records.
3. A classifier trained on QTabGAN-generated records.

All three are evaluated on the **same held-out real test records**.

A random forest is a classical prediction model made from multiple decision trees. Here it acts as a measuring tool, separate from the GAN discriminator.

```text
Real training data --------> Classifier A --+
                                           |
CTGAN synthetic data -----> Classifier B ---+--> Same real test set
                                           |
QTabGAN synthetic data ---> Classifier C ---+
```

The real-trained classifier is a useful reference, not a perfect upper bound.

If a classifier trained on synthetic records performs well on unseen real records, the generated data has preserved useful predictive relationships.

The benchmark also examines feature distributions and diversity. A small prediction-score gap alone is insufficient to establish good synthetic data.

## What do the diagnostics mean?

| Diagnostic | What it helps check |
|---|---|
| Accuracy | The fraction of test records classified correctly. It can be misleading when one class is very common. |
| F1 score | The balance between precision and recall. Macro F1 gives each class equal weight; weighted F1 weights classes by their frequency. Binary F1 focuses on the designated positive class. |
| Confusion matrix | Counts of correct predictions and mistakes for each class. Rows represent true classes and columns represent predicted classes. |
| ROC curve and AUROC | How well a binary classifier separates the classes across decision thresholds. An AUROC near 0.5 indicates chance-level ranking. |
| Precision–recall curve and average precision | Performance on the positive class, especially useful with class imbalance. Interpret it relative to the positive-class frequency. |
| GAN loss plots | How the adversarial objectives change during training. Stable or decreasing losses do not prove that the generated data is useful. CTGAN and QTabGAN use different objectives, so their loss values are not directly comparable. |
| Quantum gradient norm | Whether the training objective provides a changing signal to quantum parameters. A nonzero gradient does not guarantee useful learning. |
| CDF plots | Whether real and generated feature distributions cover similar values with similar frequencies. |
| KS statistic | A measure of distributional difference. Smaller values indicate closer distributions. |
| Jensen–Shannon divergence, or JSD | A comparison of categorical frequency distributions. Smaller values indicate closer distributions. |
| Correlation difference | Whether relationships between features are preserved. |
| Category coverage | How many categories present in training data appear in the generated data. |
| Within-class standard deviation | Whether generated records have realistic variation within each requested class. |

The paper does not specify its F1 averaging convention, so the code reports several conventions rather than assuming they are interchangeable.

High-cardinality categorical features can create hundreds of one-hot columns. Averaging diagnostics across all those columns can hide failures. The additional original-column diagnostics help avoid that problem.

## What was changed in the original code?

The initial repository already contained a quantum circuit and a basic hybrid GAN. The work extended and corrected that starting point.

### Quantum circuit and training

- Changed the default quantum width from 4 to 8 qubits to match the paper.
- Initialized trainable rotation angles uniformly over `[0, 2*pi)`.
- Added batched analytic simulation for faster training.
- Shared the gate definition between training, shot sampling, and circuit reports.
- Added finite-shot parameter-shift training.
- Added an explicit option for repeating Hadamards in each circuit layer.
- Preserved optimizer state between successive training stages.
- Added checks for invalid data and nonfinite training values.

### Evaluation and comparison

- Fixed a serious evaluation bug: the original utility functions borrowed training labels from the test-label array. They now require separate real-training and synthetic-training labels.
- Added a benchmark runner using saved train/test splits.
- Integrated the SDV CTGAN baseline.
- Added predictive, distributional, and diversity diagnostics.
- Saved configurations, library versions, dataset hashes, generated data, model checkpoints, and numerical reports. New runs also record training-source hashes.

### Supporting work

- Fixed duplicated output-column names when refitting the preprocessor.
- Updated the basic training CLI to use one-hot classification labels, including numeric class labels.
- Added circuit diagrams and a verified ideal-circuit simplification.
- Added tests, dependency records, documentation, and future noise-integration pointers.

## Does the implementation exactly reproduce the paper’s architecture?

It is a **paper-based reconstruction**, with explicit assumptions and deviations.

The main structure follows the paper: an 8-qubit RY/RZ circuit with circular CNOT entanglement, quantum probabilities concatenated with a one-hot label, a classical mapper, and a classical discriminator.

However, the paper does not provide enough implementation detail to determine every setting unambiguously.

| Setting | What was used in the recorded long runs |
|---|---|
| Circuit layers | 2; the paper does not give a numerical layer count. |
| Mapper hidden widths | `(128, 128)`; an implementation choice. |
| Discriminator hidden widths | `(128, 64, 32)`; an implementation choice. |
| Hadamards | Applied once initially. The paper’s layer wording could imply repetition; a repeated-H option is available. |
| Noise encoding | RY rotations with random angles in `[0, pi]`; the exact encoding operation is unspecified in the paper. |
| Quantum training | Analytic probabilities with simulator backpropagation. The paper describes finite-shot estimation and parameter-shift; that alternative is implemented and tested separately. |
| Numerical jitter | Disabled in the recorded benchmarks. |
| Downstream classifier | A 100-tree random forest. The paper’s downstream classifier configuration and F1 averaging are unresolved. |

Data preparation also involves assumptions. Adult retains 48,842 records after combining the public files, while the paper lists approximately 48,000. Loan removes the ID column and treats ZIP code as categorical, leaving a different predictor count from the paper’s nominal count.

Consequently, poor results from this reconstruction do not establish that the authors’ exact implementation would fail.

The detailed audit and experiment commands are in [README.md](README.md).

## What experiments were actually run?

The main experiments used two public classification datasets:

- **Adult:** demographic records with an income-category target.
- **Loan:** Universal Bank customer records with a personal-loan target.

The runs used the full prepared datasets, one seed (`42`), and 100 epochs for each generator on each dataset. QTabGAN was also evaluated at epoch 30.

The long quantum runs used analytic probabilities, backpropagation, and initialization-only Hadamards. Those settings must be kept in mind when interpreting the results.

Additional checks included a paired analytic-versus-1,024-shot generation experiment on the trained Loan model, small alternative-configuration runs, and nine unit tests.

## What were the results?

These are accuracies of the downstream random-forest classifiers on held-out real records:

| Dataset | Trained on real records | Trained on CTGAN records | Trained on QTabGAN records |
|---|---:|---:|---:|
| Adult | 85.71% | 81.84% | 64.21% |
| Loan | 98.27% | 91.47% | 57.80% |

CTGAN performed better than the tested QTabGAN configuration on both datasets.

The QTabGAN results were also below the majority-class baselines: approximately 76.07% for Adult and 90.40% for Loan. A majority-class classifier simply predicts the most common class for every record.

The paper’s reported QTabGAN accuracy gaps were not matched:

| Dataset | Paper’s real-versus-synthetic accuracy gap | This implementation’s gap |
|---|---:|---:|
| Adult | 0.0020 | 0.2150 |
| Loan | 0.0220 | 0.4047 |

A gap of `0.0020` means 0.20 percentage points; a gap of `0.2150` means 21.50 percentage points.

These are exploratory, single-seed results. The paper and local runs also have different real-trained reference scores, so matching a gap by itself would not prove exact reproduction.

See the [combined comparison plot](results/comparison.png) and [paper-comparison table](results/paper_comparison.csv).

## What went wrong with QTabGAN?

The diagnostics show **mode collapse**: the generator produces too little variety.

On Loan, generated numerical standard deviations within a class were only about 0.08%–0.35% of the real values. Average categorical coverage was approximately 38.9%. On Adult, average categorical coverage was approximately 18.1%.

In plain language, many generated records are nearly the same, and many real categories are missing from the generated tables. A classifier trained on those records cannot learn the variety of relationships present in the real data.

More epochs did not resolve the problem. On Loan, accuracy increased from 54.0% at epoch 30 to 57.8% at epoch 100, while AUROC decreased from approximately 0.728 to 0.588.

Changing the same trained Loan model from analytic generation to 1,024-shot generation also did not resolve collapse: both modes gave approximately 57.9% accuracy in the paired check. That experiment tests generation-time sampling, not finite-shot training.

The underlying cause has not been conclusively isolated. Architecture assumptions, training settings, preprocessing, mapper behavior, and label consistency need controlled investigation. Nonzero quantum gradients and passing circuit tests establish computational checks, not good synthetic-data quality.

## What does the circuit optimization do?

A separate report script creates diagrams and tests a simpler circuit for the same ideal measurement probabilities.

It merges adjacent RY rotations and removes final-layer RZ rotations whose phases cannot change the probabilities before the final computational-basis permutation.

For the 8-qubit, 2-layer circuit:

| Quantity | Original | Simplified |
|---|---:|---:|
| Gates | 64 | 48 |
| Logical depth | 22 | 20 |
| CNOT gates | 16 | 16 |
| Retained trainable angles | 32 | 24 |

The original and simplified probabilities matched to numerical precision across the tested inputs, including tests with trained checkpoints.

This preserves the ideal output probabilities, not necessarily the full quantum state. It does not establish a reduction in backend-native entangling depth or an improvement under hardware noise. The simplified circuit was not substituted into the recorded benchmark training runs.

## Which files should I read?

| File | Purpose |
|---|---|
| [benchmark.py](benchmark.py) | Runs the complete classification comparison and saves results. |
| [qtabgan/config.py](qtabgan/config.py) | Defines model and training settings. |
| [qtabgan/quantum_generator.py](qtabgan/quantum_generator.py) | Defines the quantum gates, probability calculation, and shot sampling. |
| [qtabgan/networks.py](qtabgan/networks.py) | Defines the classical mapper and discriminator networks. |
| [qtabgan/model.py](qtabgan/model.py) | Connects the components and implements adversarial training. |
| [qtabgan/preprocessing.py](qtabgan/preprocessing.py) | Encodes real data and converts generated values back into tables. |
| [qtabgan/evaluation.py](qtabgan/evaluation.py) | Contains utility and statistical evaluation helpers. |
| [scripts/prepare_data.py](scripts/prepare_data.py) | Prepares the downloaded Adult and Loan files. |
| [scripts/diagnose_tables.py](scripts/diagnose_tables.py) | Checks original-column distributions and class-specific diversity. |
| [scripts/compare_paper.py](scripts/compare_paper.py) | Compares local QTabGAN measurements with the paper’s Table 3. |
| [scripts/plot_comparison.py](scripts/plot_comparison.py) | Creates the combined utility comparison plot. |
| [scripts/evaluate_shots.py](scripts/evaluate_shots.py) | Compares analytic and finite-shot generation from a saved model. |
| [scripts/circuit_report.py](scripts/circuit_report.py) | Draws circuits and verifies the ideal simplification. |
| [tests/test_core.py](tests/test_core.py) | Checks circuit behavior, gradients, shot decoding, preprocessing, and evaluation. |
| [train_demo.py](train_demo.py) | Runs a small artificial-data demonstration. |
| [train.py](train.py) | Provides basic CSV-based training without the full benchmark evaluation. |
| [README.md](README.md) | Contains installation instructions, experiment commands, assumptions, and detailed findings. |
| [qtabgan/README_CODE_NOTES.md](qtabgan/README_CODE_NOTES.md) | Explains where future IonQ and IBM noise simulation could connect. |

A useful reading order is `config.py`, `quantum_generator.py`, `networks.py`, `model.py`, then `benchmark.py`.

## Where are the outputs?

The main results are under:

```text
results/
  comparison.png
  paper_comparison.csv
  paper_reference.json
  adult_100/
  loan_100/
```

Inside each dataset’s run directory, `summary.csv` contains model scores. The `seed_42` directory contains the saved split, preprocessing object, real-trained reference, and model-specific outputs.

Model directories contain generated records, metrics, loss plots, confusion matrices, ROC/precision–recall curves, distribution plots, and checkpoints. QTabGAN directories also contain quantum circuit reports and checkpoint evaluations.

Some local artifacts, including datasets and learned weights, are excluded from Git. Their presence on this machine does not mean they will automatically be included when the source repository is shared.

## What has been verified?

Nine unit tests passed. They check:

- Batched circuit outputs against individual circuit evaluations.
- Agreement between analytic backpropagation and parameter-shift gradients.
- Finite-shot training probabilities and gradients.
- The repeated-Hadamard architecture option.
- Bit ordering when converting shot counts into probability columns.
- Separation of training labels from test labels.
- Quantum parameter updates and optimizer preservation between training stages.
- Preprocessing refit behavior and preservation of training-fitted settings during holdout transformation.
- Correlation differences over unique feature pairs rather than the diagonal.

Small end-to-end smoke runs also passed for analytic training, finite-shot generation, repeated Hadamards, and finite-shot training.

Passing these checks means those code paths behave as checked. It does not mean the model reproduces the paper or generates useful data.

## What has not been done?

- The paper’s numerical results have not been reproduced.
- The alternative finite-shot training configuration has not received the same full 100-epoch benchmark evaluation.
- Multiple-seed uncertainty has not been established.
- Regression benchmarks have not been reproduced.
- Hardware noise experiments have not been run.
- No jobs have been submitted to IonQ or IBM quantum hardware.

## What should happen next?

First, investigate the noiseless reproduction: resolve the paper’s unspecified settings, compare architecture interpretations, and determine why the generator collapses. Use a separate validation set for tuning rather than selecting configurations based on the test set.

Once a satisfactory baseline is established, compare ideal simulation, finite-shot simulation, and hardware-noise simulation at controlled settings. Only then assess whether circuit changes improve performance on a particular device.

The integration pointers for that later phase are in [qtabgan/README_CODE_NOTES.md](qtabgan/README_CODE_NOTES.md).

## Update: code for running a checkpoint on IonQ

A hardware execution path is now implemented for direct IonQ Quantum Cloud access. It prepares the trained quantum circuits on this computer, sends them to the selected IonQ backend when you explicitly run `submit`, retrieves the measured probabilities, and uses the saved classical mapper to create CSV records locally.

The new files are `qtabgan/ionq.py` and `scripts/ionq_generate.py`. The beginner-oriented setup and execution instructions are in [IONQ_HARDWARE.md](IONQ_HARDWARE.md). Local checks and a mocked cloud round trip have been added; an authenticated cloud or hardware run has not been performed. This supports generation from fixed checkpoints, not training the model on hardware.
