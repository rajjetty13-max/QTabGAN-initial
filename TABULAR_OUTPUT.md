# Mixed-type generation and conditional moments

The previous linear-output generator produced invalid categorical vectors during training. In a retained gain-4 checkpoint, about 45% of categorical coordinates were negative, ZIP decoded to two categories, and every other categorical feature decoded to one. The discriminator saw these raw vectors, while the utility evaluator decoded and re-encoded them as valid one-hots. The new optional output adapter makes the categorical representation agree during training and evaluation.

`--tabular-output` applies `pi * sigmoid` to numerical outputs and straight-through Gumbel-softmax sampling separately to each categorical feature. The forward result is exactly one-hot; its backward pass uses the soft relaxation. Sampling occurs during training and inference. This adds classical randomness to categorical generation; increased category coverage cannot be attributed solely to the quantum circuit.

`--moment-loss-weight 10` adds a class-conditional penalty to the generator's adversarial loss. Targets are computed exclusively from the training split. The penalty matches numeric means, standard deviations and population covariances after scaling by training-wide standard deviations, plus per-feature categorical marginals. Classes present in a batch receive equal weight. Each categorical feature contributes one vector error, so the hundreds of ZIP indicators do not become hundreds of independent feature penalties. Batch classes with fewer than two rows skip variance/covariance estimation.

These are explicit extensions to the supplied paper's linear mapper and adversarial objective. The quantum probability input, circuit, mapper widths and discriminator remain the same as the gain-4 control. Moment matching does not guarantee matching complete distributions or recovery from mathematically exact collapse. Default settings retain the original linear output and loss.

## Completed comparison

[Results, individual seeds and diagnostics](results/tabular_moment_study/analysis/REPORT.md) compare three groups: gain-4 control; mixed-type output alone; and mixed-type output with moment weight 10. Each uses 30 epochs, seeds 42/43/44, the same paired 60/20/20 splits, two-layer `paper_readout` circuit, D(x,y), batch size 64, and G/D learning rates 0.0002. Checkpoints are selected by validation macro F1. Test scores are computed after each run and are not used to rank variants.

The moment candidate raised mean validation macro F1 **0.5735 → 0.7121**. Its selected checkpoints raised mean test accuracy **77.0% → 85.2%**, macro F1 **0.6013 → 0.6971**, AUROC **0.7923 → 0.8764**, AP **0.3563 → 0.4093**, and minority recall **54.9% → 71.5%**. The average numeric within-class standard-deviation ratio increased **1.26% → 57.56%**; category coverage increased **40.8% → 96.7%**. Mean test macro F1 also improved at the shared final epoch, although training still fluctuates. Seed 42's selected test F1 decreased slightly while the other two improved; this is a mean improvement, not a gain on every seed.

Mixed-type output alone restored categorical coverage but left numeric variation near zero and harmed AUROC. Retain the moment penalty with the output adapter. This experiment does not establish that each component is necessary: a moment-only linear-head ablation was not run. All three control loss histories exactly reproduced the previous gain-4 study.

A [separate classifier-threshold audit](results/tabular_moment_study/threshold_calibration/REPORT.md) froze these checkpoints and chose thresholds using validation macro F1 only. Applying that procedure to both old and new models gave mean test accuracy **83.9% → 86.0%** and macro F1 **0.6409 → 0.6978**. For the new model, calibration added only 0.8 accuracy points while reducing minority recall from 71.5% to 66.3%. The original decision rule remains the main reported protocol; calibration changes the downstream classifier, not generator quality. AUROC/AP are unchanged by threshold choice.

Loan's majority-class reference is 90.4% accuracy. Use minority recall, macro F1, AUROC, AP and numeric diversity alongside accuracy. These splits have appeared in earlier experiments, so this comparison is exploratory. The split and evaluator differ from the paper's incompletely specified protocol.

## Continuation through 100 epochs

[The completed 100-epoch comparison](results/tabular_moment_100_study/analysis/REPORT.md) continues the same three moment candidates from epoch 30, retaining weights, Adam states, RNG states and data splits. Exact loss and validation prefixes, source hashes and optimizer step counts were verified.

| Mean test metric | Selected through epoch 30 | Selected through epoch 100 |
| --- | ---: | ---: |
| Accuracy | 85.20% | 90.97% |
| Macro F1 | 0.6971 | 0.7538 |
| AUROC | 0.8764 | 0.8932 |
| Average precision | 0.4093 | 0.5229 |
| Minority recall | 71.53% | 59.72% |

Validation selected epochs 80, 90 and 70 for seeds 42, 43 and 44. Their test accuracies are 90.9%, 91.3% and 90.7%. Numeric within-class variation improved from 57.6% to 72.7% of the real reference; categorical coverage stayed around 96%. These gains come with lower minority recall. The actual final epoch 100 averages 89.73% accuracy and 0.7042 macro F1, so use the saved validation-selected checkpoints. No new classifier threshold tuning was applied. The earlier reused-split and paper-extension limitations still apply.

## Run the moment candidate

Choose a fresh output directory. The runner infers the numeric/category block widths from its train-fitted preprocessor and saves them in each checkpoint.

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper_readout --condition-discriminator --layers 2 \
  --seed 42 --classical-init-seed 42042 \
  --epochs 100 --eval-every 10 --batch-size 64 --quantum-init-gain 4 \
  --tabular-output --moment-loss-weight 10 \
  --output results/tabular_moment_next
```

For the output-only ablation omit `--moment-loss-weight`. For the original control also omit `--tabular-output`. Repeat with seeds 43/44 and classical initialization seeds 42043/42044.

The exact experiment source is retained in `results/tabular_moment_study/source/` and checked against the recorded launch hashes by:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python scripts/summarize_tabular_moments.py \
  --study results/tabular_moment_study --output results/tabular_moment_summary_next
```

For local inference, load a checkpoint using its saved configuration and state dictionary, then call `model.generate(conditions)`. Hard categorical samples remain stochastic; set the Torch seed when reproducible samples are needed. Legacy mapper diagnostics and the legacy IonQ export path support the original linear head only.

Tests cover valid output support, gradients through categorical samples into the quantum circuit, class-specific moment targets, covariance errors, finite degenerate-case gradients, unchanged legacy checkpoints, and exact optimizer/RNG continuation with the new objective.

The full suite passed **47 tests**. The archived source precedes three later fixes that do not affect this Loan protocol: warnings for purely categorical moment targets, schema validation for non-integer dimensions, and the training-loss plot label. Legacy hardware/diagnostic scripts now reject the optional mixed-type head before producing misleading raw-mapper outputs.
