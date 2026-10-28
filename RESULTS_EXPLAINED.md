# Explaining every plot, the pipeline, and the measured results

**Historical results:** The older result folders described below were removed during cleanup. Their numerical discussion is retained as history; current epoch 1–30 reports and images are indexed in [results/README.md](results/README.md).


This report explains the saved results in plain language. It uses the measurements already on disk; no models were retrained and no cloud jobs were submitted to create it.

**Main finding:** the computations finished, but the tested QTabGAN configuration produces poor synthetic data. CTGAN performs better in both tested datasets. These experiments do not reproduce the paper's QTabGAN results.

## 1. What the project is trying to achieve

The goal is to create fictional spreadsheet rows that preserve useful patterns in real data. A generator should learn both realistic individual values and relationships between columns. Copying the requested class frequencies alone is not enough.

There are two generators: CTGAN is a classical tabular generator; QTabGAN combines a simulated quantum circuit with a classical mapper. The main experiments used 100 epochs, one random seed (42), and the full prepared Adult and Loan datasets.

```text
REAL DATA
    -> split into training and test rows
    -> fit preprocessing using training rows only
    -> train CTGAN and QTabGAN on training data
    -> generate synthetic training tables

QTabGAN GENERATION
random angles -> 8-qubit circuit -> 256 probabilities
    -> append requested one-hot class -> classical mapper -> synthetic feature values
    -> inverse preprocessing -> decoded CSV records

UTILITY EVALUATION
real training rows ------> random forest A --+
CTGAN synthetic rows ----> random forest B --+--> same real test rows
QTabGAN synthetic rows --> random forest C --+
```

The classifiers used to measure usefulness are separate from the GAN discriminator. The GAN discriminator learns real-versus-fake feedback during training. The random forests measure whether a generated table can teach a useful prediction rule afterward.

The long QTabGAN runs used analytic circuit probabilities and simulator backpropagation. Finite-shot parameter-shift training and repeated Hadamards are tested code options, but were not used for these long runs. The layer count, classical widths, noise encoding and evaluator settings involve documented assumptions. See the [architecture audit](README.md#architecture-and-assumptions).

## 2. Data sizes and class balance

| Dataset | Training rows | Test rows | Positive test rows | Positive test frequency |
|---|---|---|---|---|
| Adult | 39,073 | 9,769 | 2,338 | 23.93% |
| Loan | 3,500 | 1,500 | 144 | 9.60% |


Adult's positive class is income `>50K`; its negative class is `<=50K`. Loan uses `PersonalLoan=1` as the positive class and `0` as the negative class.

Always predicting the majority class achieves **76.07% accuracy on Adult** and **90.40% on Loan**. These simple baselines are essential when reading accuracy numbers.

## 3. The main comparison image

Historical figure: Combined utility comparison (artifact removed).

Open full-size comparison (historical artifact removed). `results/loan_comparison.png` repeats the Loan comparison alone.

Each row is a dataset. The columns show accuracy, macro F1 and AUROC. Grey means a classifier trained on real records, blue means trained on CTGAN records, and orange means trained on QTabGAN records. Every bar is evaluated on the same held-out real test set for its dataset. Higher bars are better for these three metrics.

The grey reference is useful but is not a mathematical upper bound. These are measured results for one split and one seed, not averages over repeated experiments.

| Dataset | Training source | Accuracy | Macro F1 | AUROC | Average precision |
|---|---|---|---|---|---|
| Adult | real | 85.71% | 0.7924 | 0.9038 | 0.7688 |
| Adult | ctgan | 81.84% | 0.7368 | 0.8650 | 0.6466 |
| Adult | qtabgan | 64.21% | 0.4639 | 0.4962 | 0.2310 |
| Loan | real | 98.27% | 0.9456 | 0.9968 | 0.9819 |
| Loan | ctgan | 91.47% | 0.7446 | 0.8915 | 0.5473 |
| Loan | qtabgan | 57.80% | 0.4427 | 0.5877 | 0.1578 |


Accuracy is the fraction classified correctly at the classifier's prediction rule. Macro F1 balances precision and recall while giving both classes equal weight. AUROC measures ranking across thresholds: about 0.5 is chance-level ranking. Average precision summarizes the precision–recall curve; its baseline depends on the positive-class frequency.

QTabGAN accuracy is below the majority-class baseline on both datasets. Adult QTabGAN AUROC is 0.4962, essentially chance-level ranking. CTGAN is stronger here, but still falls below the real-trained reference.

## 4. How to read every `confusion.png`

A confusion matrix counts predictions. **Rows are true classes; columns are predicted classes.** Its diagonal contains correct predictions and its other two cells contain errors.

```text
                         Predicted negative   Predicted positive
True negative class             TN                   FP
True positive class             FN                   TP
```

- TN: a negative-class record correctly predicted negative.
- FP: a negative-class record incorrectly predicted positive.
- FN: a positive-class record missed by the classifier.
- TP: a positive-class record correctly predicted positive.

These are downstream prediction counts, not the discriminator's real/fake decisions and not counts of generated rows.

Historical figure: Loan QTabGAN confusion matrix (artifact removed).

| Dataset | Training source | TN | FP | FN | TP | Positive precision | Positive recall |
|---|---|---|---|---|---|---|---|
| Adult | real | 6914 | 517 | 879 | 1459 | 73.84% | 62.40% |
| Adult | ctgan | 6717 | 714 | 1060 | 1278 | 64.16% | 54.66% |
| Adult | qtabgan | 5953 | 1478 | 2018 | 320 | 17.80% | 13.69% |
| Loan | real | 1356 | 0 | 26 | 118 | 100.00% | 81.94% |
| Loan | ctgan | 1298 | 58 | 70 | 74 | 56.06% | 51.39% |
| Loan | qtabgan | 803 | 553 | 80 | 64 | 10.37% | 44.44% |


For Loan QTabGAN, the classifier correctly predicts 803 negatives and 64 positives, but makes 553 false-positive and 80 false-negative predictions. That is **867 correct and 633 wrong out of 1,500**. Only **10.37%** of its positive predictions are correct; it finds **44.44%** of actual positives.

For Adult QTabGAN, it finds only **320 of 2,338** actual higher-income records, a recall of **13.69%**. This illustrates why accuracy alone is insufficient.

## 5. How to read every `auc_curves.png`

Historical figure: Adult CTGAN ROC and precision–recall curves (artifact removed).

The left plot is the ROC curve. Its horizontal axis is the false-positive rate and its vertical axis is the true-positive rate (recall). Moving toward the upper-left corner is better. The diagonal dashed line represents chance-level ranking. The AUROC in the legend summarizes the curve, not the fraction of correct predictions.

The right plot is the precision–recall curve. Its horizontal axis is recall: how many actual positives are found. Its vertical axis is precision: how many positive predictions are correct. The horizontal dashed line marks the positive-class frequency in the real test set. The legend reports **AP (average precision)**, not trapezoidal PR-AUC.

On Loan, positive frequency is only 9.60%, so precision–recall behavior matters particularly strongly. CTGAN AP is 0.5473 versus QTabGAN AP 0.1578. On Adult, the corresponding values are 0.6466 versus 0.2310; the Adult positive frequency is 23.93%.

The `real/auc_curves.png` files are the real-trained classifier references. Files under generator folders describe classifiers trained on that generator's synthetic data.

## 6. How to read every `losses.png`

Historical figure: Adult QTabGAN losses and quantum gradients (artifact removed).

QTabGAN has two panels. The left panel shows generator and discriminator training losses over epochs. These are optimization objectives, not accuracy percentages. The right panel shows the average norm (size) of gradients on the quantum parameters. A finite, nonzero gradient confirms that a training signal reaches quantum parameters; it does not prove that the signal produces useful data.

The Adult curve's generally high generator loss and low discriminator loss are consistent with a discriminator that can readily distinguish generated records and a struggling generator. The downstream tests and distribution plots provide the actual evidence of poor data quality.

| Dataset | Stage | Generator loss | Discriminator loss | Quantum gradient norm |
|---|---|---|---|---|
| Adult | First epoch | 0.761826 | 0.580987 | 0.000333 |
| Adult | Epoch 100 | 4.712799 | 0.019678 | 0.117520 |
| Loan | First epoch | 0.673895 | 0.676953 | 0.000024 |
| Loan | Epoch 100 | 3.182703 | 0.110300 | 0.023814 |


CTGAN `losses.png` has one panel. Its generator and discriminator use different objectives from this QTabGAN implementation. Negative CTGAN discriminator loss is not negative accuracy and is not automatically an error. Do not compare the two models' loss magnitudes to decide which generator is better.

CTGAN labels epochs 0–99; QTabGAN labels them 1–100. Both long runs contain 100 epochs.

## 7. How to read `cdf.png` and `original_cdf.png`

CDF means cumulative distribution function. For a value on the horizontal axis, the vertical axis shows the fraction of records at or below that value. For example, a CDF value of 0.8 at age 50 means 80% of records have age at most 50.

Blue is real training data; orange is synthetic data. Close curves indicate similar one-dimensional distributions. A narrow, nearly vertical orange step means most generated values sit in a very small range. A few separated steep steps can mean only a few modes were generated.

`cdf.png` uses the model's transformed representation, including training-range clipping and one-hot columns. `original_cdf.png` uses original numerical columns and units, before generated values are clipped for comparison. The original-scale plots expose invalid values that transformed/clipped plots can hide. Each image displays up to six features; the CSV diagnostics cover more columns where present.

Historical figure: Loan QTabGAN original-scale CDFs (artifact removed).

In Loan QTabGAN, generated ages span only **47.95–48.73**, incomes **66.58–67.60 in the dataset's units**, and mortgage values **61.55–64.38**. That is a tiny range compared with the real table. Orange curves are nearly vertical because many generated records are almost identical.

Historical figure: Adult QTabGAN original-scale CDFs (artifact removed).

Adult has invalid values as well as collapsed variation. From the saved synthetic CSV:

| Adult generated feature | Minimum | Maximum | Negative records | Negative share |
|---|---|---|---|---|
| age | -14.92 | 8.54 | 29,724 | 76.07% |
| hours-per-week | -8.12 | 2.03 | 29,724 | 76.07% |
| capital-loss | -839.82 | 17.20 | 29,948 | 76.65% |


All 39,073 generated Adult ages are below 9 years, and 29,724 are negative. The classifier metrics alone do not reveal this. The mapper has a linear, unconstrained numerical output and the inverse transform does not enforce physical validity. That is an implementation limitation consistent with the observed invalid values, not a conclusive explanation of every training failure. These records have not been repaired in the saved experiment.

Distribution diagnostics quantify the differences:

| Dataset | Model | Numerical KS ↓ | Categorical JSD ↓ | Category coverage ↑ | Numerical correlation error ↓ |
|---|---|---|---|---|---|
| Adult | ctgan | 0.1919 | 0.0064 | 100.00% | 0.0246 |
| Adult | qtabgan | 0.8590 | 0.4926 | 18.09% | 0.9998 |
| Loan | ctgan | 0.1684 | 0.0234 | 94.26% | 0.1083 |
| Loan | qtabgan | 0.5729 | 0.2324 | 38.92% | 0.6214 |


Smaller KS, JSD and correlation error indicate closer distributions; higher coverage means more observed training categories were generated. Coverage is averaged over categorical columns and is not the fraction of whole records that are correct. JSD uses natural logarithms here; it is not a percentage error. Correlation error averages unique numerical feature pairs and excludes columns constant in either table.

The one-hot-averaged summaries can be misleading. Loan QTabGAN has a transformed-column mean KS of **0.0181**, but its original numerical-column mean KS is **0.5729**. Hundreds of ZIP-code indicator columns dominate the first average. They do not establish good numerical fidelity. Loan has **473 constant synthetic transformed columns**; Adult has **101**. A constant one-hot column means its category never appears or always appears, not that there are 473 original numerical columns.

Within each requested class, Loan numerical standard deviations are only approximately **0.08%–0.35%** of the real values. This is strong evidence of mode collapse. A correct class-frequency ratio does not guarantee realistic records within those classes.

## 8. What the epoch directories mean

`epoch_30/` saves the earlier QTabGAN evaluation. `epoch_100/` saves the final checkpoint evaluation. The top-level model plots represent the final evaluation. Thus some PNGs repeat the same final result in two locations.

| Dataset | QTabGAN epoch | Accuracy | Macro F1 | AUROC |
|---|---|---|---|---|
| Adult | 30 | 36.73% | 0.3212 | 0.3259 |
| Adult | 100 | 64.21% | 0.4639 | 0.4962 |
| Loan | 30 | 54.00% | 0.4585 | 0.7285 |
| Loan | 100 | 57.80% | 0.4427 | 0.5877 |


Extra epochs raised accuracy in these runs but did not establish successful generation. Loan macro F1 and AUROC worsened. Adult improved from its earlier checkpoint but still failed the majority baseline and had invalid generated values.

These checkpoints inspect the test set for diagnosis. They should not be used to select the best future hyperparameters; that requires a separate validation set.

## 9. What the paired-shot images mean

The folders under `loan_100/seed_42/qtabgan/paired_1024_shots/` repeat confusion, CDF and ROC/PR plots for two generation modes from the same trained checkpoint and identical latent angles:

| Mode | Accuracy | AUROC | Average precision |
|---|---:|---:|---:|
| Analytic generation | 57.87% | 0.6051 | 0.1682 |
| 1,024-shot generation | 57.87% | 0.5939 | 0.1595 |

Both confusion matrices are `[[804, 552], [80, 64]]`. Shot sampling did not resolve the failure. This comparison tests generation-time sampling; the checkpoint itself was trained analytically. Its numbers differ slightly from the main Loan evaluation because the paired experiment uses a different fixed set of latent angles.

## 10. How to read circuit images

Historical figure: Simplified reference circuit (artifact removed).

Read the circuit from left to right. Each horizontal line is one qubit; 0–7 means eight qubits. H is a Hadamard gate. RY and RZ rotate a qubit by trainable angles. A filled control dot connected to a circled plus is a CNOT: it conditionally flips the target. The long last connection closes the directed ring. The meter symbols indicate the final computational-basis measurement.

`circuit.png` shows the model's circuit topology. `original.png` and `merged.png` compare its original and simplified forms. Pictures generally label gate types rather than exposing every full-precision angle; saved checkpoint tensors contain the numerical weights. This is a circuit diagram, not a plot of model accuracy or physical hardware layout.

| Quantity | Original | Simplified |
|---|---:|---:|
| Qubits | 8 | 8 |
| Possible measurement outcomes | 256 | 256 |
| Gates | 64 | 48 |
| Logical depth | 22 | 20 |
| CNOTs | 16 | 16 |
| Retained trainable angles | 32 | 24 |

The simplification merges adjacent RY rotations and removes last-layer RZ angles that cannot affect the final ideal measurement probabilities. Gate count falls **25%** and logical depth about **9.09%**. CNOT count does not fall.

For eight tested reference inputs, maximum probability error was approximately **1.25e-16**. Trained Adult and Loan checkpoint checks were also below `1e-16`. This is agreement to floating-point precision for those tests, supported by the gate identities used. It does not prove identical noisy-device behavior or reduced native hardware depth.

`results/circuit/` uses a reference parameter initialization. The `optimized_circuit/` folders under the Adult and Loan QTabGAN results use the corresponding trained checkpoints. Their topology can look identical while their angle values differ. The simplification was not substituted into the recorded 100-epoch training runs.

## 11. Why there are smoke-test images

`smoke`, `finite_shot_smoke`, `repeated_h_smoke` and `finite_training_smoke` are small software checks using the bundled breast-cancer dataset. They exercise code paths, not the Adult/Loan paper benchmarks. Their plots have the same meanings described above but use different sample counts, circuit sizes and epoch counts. They should not be compared as if they were repeated full-data experiments.

## 12. How to interpret the paper comparison

The paper's Table 3 reports QTabGAN examples for Adult and Loan. It does not provide a CTGAN target in that table.

| Dataset | Paper accuracy gap | Local QTabGAN accuracy gap | Local CTGAN accuracy gap |
|---|---:|---:|---:|
| Adult | 0.0020 (0.20 percentage points) | 0.2150 (21.50 points) | 0.0387 (3.87 points) |
| Loan | 0.0220 (2.20 percentage points) | 0.4047 (40.47 points) | 0.0680 (6.80 points) |

The gap is the absolute difference between the real-trained and synthetic-trained classifiers' accuracies. The paper's F1 gaps are 0.0080 and 0.0279, but its F1 averaging convention is unresolved. Our QTabGAN macro-F1 gaps are 0.3285 and 0.5029. Those conventions cannot be assumed identical.

Both local QTabGAN accuracy-gap checks fail. This does not establish that the authors' exact implementation fails: preprocessing, classifier settings and aspects of the architecture/training differ or are unspecified. One seed provides no reliable run-to-run uncertainty estimate. No quantum advantage is demonstrated.

Recorded workflow elapsed times were about 37.11 minutes (Adult QTabGAN), 13.49 minutes (Adult CTGAN), 5.48 minutes (Loan QTabGAN), and 7.05 minutes (Loan CTGAN). These include diagnostics and evaluation, including the extra QTabGAN checkpoints, so they are not comparable pure-training benchmarks.

## 13. How `qtabgan/ionq.py` fits into the project

The IonQ path executes a fixed checkpoint's quantum circuit remotely, then runs its classical mapper locally. It does not retrain the model or repair its learned distribution.

| Function | Role |
|---|---|
| `export_qis` | Reads the existing PennyLane gate tape and exports IonQ-compatible H, RY, RZ and CNOT operations. |
| `simulate_qis` | Replays that exported JSON locally to check that export did not change the calculation. |
| `probabilities_from_ionq` | Fills sparse missing outcomes with zeros and converts IonQ's decimal outcome ordering into the mapper's ordering. |
| `IonQClient` | Sends authenticated HTTP requests to direct IonQ Quantum Cloud. |
| `load_bundle` | Checks that the prepared plan and copied model/preprocessor have not changed. |
| `submit_bundle` | Journals each submission and saves accepted job IDs; it does not automatically retry POSTs. |
| `attach_job` | Recovers an accepted job ID after a lost response, without creating another job. |
| `collect_bundle` | Checks saved jobs and retrieves completed results, keeping rows in their original order. |

`scripts/ionq_generate.py` connects those functions to the frozen mapper and CSV decoder. Its commands are `prepare`, `local`, `backends`, `submit`, `collect`, and `attach`. See [IONQ_HARDWARE.md](IONQ_HARDWARE.md) for executable instructions.

In the prepared local Loan bundle, **4 circuits × 1,024 shots = 4,096 requested circuit-shots**. The exported probabilities matched the checkpoint with maximum error about **2.68e-8**. Local shot sampling produced mean total-variation distance **0.1421** from the ideal quantum distributions, but mean absolute change in mapper feature coordinates was only **0.0000858**. This measures a small pilot's response to sampling, not predictive accuracy or device robustness.

No live IonQ jobs or hardware-result plots have been produced. Existing plots show local experiments. Nineteen tests passed in the prior validation, including mocked API and full mapper/CSV round-trip tests; that does not substitute for an authenticated cloud test.

## 14. What is established, and what remains uncertain?

The circuit/export/sampling and evaluation paths have software checks. The observed QTabGAN tables have low utility, missing categories, collapsed within-class variation, and invalid Adult values. CTGAN performs better under the tested settings.

The precise cause of QTabGAN's failure is not isolated. Candidate issues include unconstrained mapper outputs, conditioning that the discriminator cannot directly verify, preprocessing choices, optimizer settings, and paper ambiguities. These are hypotheses to test, not demonstrated root causes. Adding hardware noise has not been shown to improve any of them.

## 15. Index of all saved images

The index below covers every PNG found under `results/` when this report was written. Repeated filenames are separated by dataset, generator, checkpoint or smoke-test experiment.


Total images: **99**.


| Image | What it means |
|---|---|
| results/adult_100/seed_42/ctgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/adult_100/seed_42/ctgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. |
| results/adult_100/seed_42/ctgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/adult_100/seed_42/ctgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. |
| results/adult_100/seed_42/ctgan/original_cdf.png (historical artifact removed) | Original numerical units, exposing collapse and invalid values; section 7. |
| results/adult_100/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/adult_100/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. |
| results/adult_100/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. |
| results/adult_100/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/adult_100/seed_42/qtabgan/epoch_100/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/epoch_100/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/epoch_100/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/epoch_30/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/epoch_30/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/epoch_30/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. |
| results/adult_100/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. |
| results/adult_100/seed_42/qtabgan/optimized_circuit/merged.png (historical artifact removed) | Simplified ideal-probability-equivalent circuit; section 10. |
| results/adult_100/seed_42/qtabgan/optimized_circuit/original.png (historical artifact removed) | Original reference or trained-checkpoint circuit; section 10. |
| results/adult_100/seed_42/qtabgan/original_cdf.png (historical artifact removed) | Original numerical units, exposing collapse and invalid values; section 7. |
| results/adult_100/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/adult_100/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/circuit/merged.png (historical artifact removed) | Simplified ideal-probability-equivalent circuit; section 10. |
| results/circuit/original.png (historical artifact removed) | Original reference or trained-checkpoint circuit; section 10. |
| results/comparison.png (historical artifact removed) | Main two-dataset utility comparison; section 3. |
| results/finite_shot_smoke/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_1/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_1/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_1/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_2/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_2/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/epoch_2/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/finite_shot_smoke/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/epoch_1/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/epoch_1/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/epoch_1/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/finite_training_smoke/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/loan_100/seed_42/ctgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/loan_100/seed_42/ctgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. |
| results/loan_100/seed_42/ctgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/loan_100/seed_42/ctgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. |
| results/loan_100/seed_42/ctgan/original_cdf.png (historical artifact removed) | Original numerical units, exposing collapse and invalid values; section 7. |
| results/loan_100/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/loan_100/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. |
| results/loan_100/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. |
| results/loan_100/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/loan_100/seed_42/qtabgan/epoch_100/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/epoch_100/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/epoch_100/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/epoch_30/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/epoch_30/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/epoch_30/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. |
| results/loan_100/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. |
| results/loan_100/seed_42/qtabgan/optimized_circuit/merged.png (historical artifact removed) | Simplified ideal-probability-equivalent circuit; section 10. |
| results/loan_100/seed_42/qtabgan/optimized_circuit/original.png (historical artifact removed) | Original reference or trained-checkpoint circuit; section 10. |
| results/loan_100/seed_42/qtabgan/original_cdf.png (historical artifact removed) | Original numerical units, exposing collapse and invalid values; section 7. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/analytic/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Paired generation experiment; section 9. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/analytic/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Paired generation experiment; section 9. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/analytic/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Paired generation experiment; section 9. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/finite_shot/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Paired generation experiment; section 9. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/finite_shot/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Paired generation experiment; section 9. |
| results/loan_100/seed_42/qtabgan/paired_1024_shots/finite_shot/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Paired generation experiment; section 9. |
| results/loan_100/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. |
| results/loan_100/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. |
| results/loan_comparison.png (historical artifact removed) | Loan-only utility comparison; section 3. |
| results/repeated_h_smoke/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/epoch_2/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/epoch_2/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/epoch_2/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/repeated_h_smoke/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/ctgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/ctgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/ctgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/ctgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/circuit.png (historical artifact removed) | Model quantum circuit topology; section 10. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/epoch_2/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/epoch_2/cdf.png (historical artifact removed) | Up to six features in transformed/clipped model coordinates; section 7. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/epoch_2/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Earlier/final checkpoint snapshot; section 8. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/qtabgan/losses.png (historical artifact removed) | Training losses; QTabGAN also shows quantum gradient norms; section 6. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/real/auc_curves.png (historical artifact removed) | ROC/AUROC and precision–recall/AP; section 5. Small software check, not a paper benchmark; section 11. |
| results/smoke/seed_42/real/confusion.png (historical artifact removed) | Downstream true-versus-predicted class counts; section 4. Small software check, not a paper benchmark; section 11. |
