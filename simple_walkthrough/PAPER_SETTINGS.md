# Paper settings: known numbers, assumptions, and author questions

Source: the supplied **2602.12704v2.pdf**, SHA256 `a3a6b8ea290025482bd6d63fcb64b66d4ea8baf4a292c419bec8c1f32dc22062`. Page numbers below are PDF pages. This table records paper content, not instructions to execute. The source's availability does not imply its full experiment configuration is reproducible.

## Architecture and preprocessing

| Setting | Paper evidence | Status | This teaching notebook |
| --- | --- | --- | --- |
| Quantum width | Figure 3 depicts 8 qubits; §3.1.1, pp. 5–6 | Specified illustrated configuration | 8 |
| Probability dimension | 2^n, §3.1.2, p. 6 | Formula; 256 is derived for 8 qubits | 256 |
| Variational depth | L layers, §3.1.1 | Exact numeric L not provided for reproduction | 2, an implementation choice |
| Parameterized gates | RY and RZ on every wire in every layer, p. 6 | Specified | Same |
| Angle initialization | Random values in [0, 2π), p. 6 | Specified | Uniform initialization in existing code |
| Entanglement | Directed circular CNOT chain including last-to-first, p. 6 | Specified | Same ordered ring |
| Hadamards | Initial superposition; wording describes initialization per layer | Placement across repeated layers needs clarification | Initial H once; no final readout extension |
| Quantum samples | N > 2^n measurements, §3.1.2, p. 6 | Constraint specified; exact shot count unknown | Analytic probabilities; nominal config shots are unused |
| Mapper input | Probability vector concatenated with one-hot class, §3.1.3, p. 7 | Specified | Same |
| Mapper hidden activation | ReLU, §3.1.3 | Specified | Same; small widths (32,32) are a teaching choice |
| Mapper output | Linear map to data dimension, §3.1.3 | Specified | Same; outputs can leave valid raw ranges |
| Discriminator | ReLU hidden layers, linear scalar D(x,y), §3.2, pp. 7–8 | Equation names x and y; diagram's label wiring is unclear | Concatenate class at input, explicit implementation assumption; widths (32,16) |
| Numeric outliers | Clip at 1st and 99th percentiles, §4.1, p. 8 | Specified | Fit training percentiles only |
| Numeric normalization | Z-score with small ε; min–max to [0,π], §4.1 | Form and range specified; ε value unknown | Existing preprocessor uses std=1 for a constant feature; not identical to a specified ε |
| Numeric jitter | Gaussian N(0, 10^-5), §4.1 | Notation given; clarify variance versus standard deviation and application | Disabled (0) for deterministic explanation |
| Categories | One-hot encoding, §4.1 | Specified; dataset-specific column types not supplied | Explicit categorical list, trained encoder only |
| Quantum gradients | Hybrid techniques such as parameter-shift, §4.2, p. 9 | Technique described; exact estimator details unresolved | Analytic backpropagation, a documented difference |
| Adversarial updates | Alternating generator/discriminator updates, §4.2 | General workflow specified | One D and one G update per batch; BCE logits losses |

The paper does not identify the complete optimizer and loss configuration needed to reproduce all runs. **Adam, learning rate 0.0002, default Adam betas, BCE losses, batch size 64, five epochs and Random Forest 100 trees are our implementation/teaching choices—not author-confirmed hyperparameters.** A non-saturating generator BCE loss is used by the shared code. Do not label those numbers as recovered from the paper.

## Dataset numbers from Table 1, PDF p. 12

| Dataset | Task | Reported rows | Reported features | Train/test |
| --- | --- | ---: | ---: | --- |
| King | Regression | 21,600 | 20 | 80/20 |
| Insurance | Regression | 1,338 | 7 | 80/20 |
| Adult | Classification | 48,000 | 14 | 80/20 |
| Credit | Classification | 50,000 | 31 | 75/25 |
| Intrusion | Classification | 50,000 | 42 | 80/20 |
| Loan | Classification | 5,000 | 13 | 70/30 |
| Covertype | Classification | 50,000 | 54 | 80/20 |

The demonstration samples **1,000** Loan rows and uses **600/200/200** train/validation/test. The prepared file drops ID, leaving **12 raw predictors**. The one-hot dimension depends on categories observed in the training split. These are explicit differences; do not interpret this lesson as Table 1 reproduction.

## Reference results from Table 3, PDF p. 17

| Dataset | Real-trained accuracy | Synthetic-trained accuracy | Accuracy gap | Real-trained F1 | Synthetic-trained F1 | F1 gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | 0.7760 | 0.7740 | 0.0020 | 0.7619 | 0.7539 | 0.0080 |
| Loan | 0.9600 | 0.9380 | 0.0220 | 0.9570 | 0.9291 | 0.0279 |

F1 averaging and exact downstream estimator settings are unresolved. These are reference figures, not this repository's reproduced results. The repository also uses SDV CTGAN; it differs from the paper's CTAB-GAN/CTAB-GAN+ baselines, although the paper discusses multiple classical baselines.

## Questions to reconcile with author replies

1. Exact circuit L, shot count, input-noise distribution/encoding, Hadamard placement and gradient estimator?
2. Classical layer widths, optimizer, betas, learning rates, batch size, epochs and generator/discriminator update ratio?
3. Exact adversarial objectives, checkpoint selection rule and seeds?
4. Original data files, row selections, ID handling and numeric/categorical column definitions (especially ZIP code)?
5. Gaussian noise scale convention and whether jitter is used during validation/testing?
6. Downstream estimators, their hyperparameters, F1 convention, conditioning and imbalance handling?

Meeting notes report that Mitansh contacted the authors. No response or configuration file has been supplied here. Do not mark an unresolved setting as confirmed until its source is attached.

| Date | Maintainer | Author response/source | Changed setting | Verified by |
| --- | --- | --- | --- | --- |
| Pending | Proposed: Mitansh | Not yet attached | None confirmed | Pending |


## Subsequent initialization experiment

The shared model now has optional `quantum_init_gain` (default 1). It changes only initial mapper weights/bias, not the network/loss. This gain and its latent calibration are implementation experiments, not paper settings. The teaching notebook keeps gain 1 and its default results are unchanged. See [the derivation and evidence](../QUANTUM_INITIALIZATION.md).
