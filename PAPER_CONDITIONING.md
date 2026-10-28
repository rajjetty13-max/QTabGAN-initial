# Testing the paper's label-conditioned discriminator

The earlier circuit experiment improved logical depth and removed inactive parameters, but did not improve downstream predictive utility. This experiment returns to the original circuit and checks the conditioning path against the supplied paper, `2602.12704v2.pdf`.

## What the paper supports

Section 3.1.3 defines the classical mapper's input as the concatenation of the quantum probability vector and a one-hot class label. Its hidden layers use ReLU and its final layer is linear. These remain unchanged.

Section 3.2 explicitly writes the discriminator as **D(x, y)** and identifies y as the corresponding class label. Figure 4 and the training-workflow prose do not clearly show how y enters the network. The previous implementation followed the feature-only diagram. Calling that the sole paper interpretation was incomplete.

The new option concatenates the one-hot label with features at the discriminator input. This is a concrete implementation of the equation's conditioning, not an input-wiring detail explicitly specified by the paper. Both real and generated rows receive their corresponding labels, including the discriminator call used to train the generator.

The original quantum ansatz, mapper activations, linear output, preprocessing, adversarial BCE losses and model widths are retained. No auxiliary classifier, output clipping layer, feature-matching loss, or replacement generative architecture is introduced.

## Compatibility and use

`QTabGANConfig(condition_discriminator=True)` or `--condition-discriminator` enables D(x,y). The default is false to preserve historical checkpoint shapes and behavior. Old checkpoints lack this setting and continue to load with the original feature-only discriminator. New checkpoint configurations save the choice explicitly.

Changing the setting changes the discriminator's input width. Train a new model for this comparison; do not attach an old discriminator's weights to the new shape.

## Matched experiment

Each variant trains for 30 epochs, with evaluations at 1, 10, 20 and 30. The experiment uses the original eight-qubit/two-layer circuit and the full Loan dataset. For each seed (42, 43, 44), the control and conditioned variant share the same stratified 60/20/20 train/validation/test split, batch size 64, classical generator initialization, learning rates and preprocessing. Checkpoints are selected by validation macro F1; test results are reported after training.

The fresh training RNG is set to seed + 20000 after model initialization. This prevents the changed discriminator size from shifting minibatch and latent-noise draws. It also means this control is a new experiment; compare it with its matched conditioned run, not with the previous circuit trial's numbers. Across seeds, the split and initialization both vary, so the spread is not an initialization-only uncertainty estimate.

Example command (use a fresh output directory):

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper --condition-discriminator --seed 42 \
  --epochs 30 --eval-every 10 --batch-size 64 \
  --output results/conditioning_next
```

Run the matched control without `--condition-discriminator`. `scripts/compare_training_runs.py` verifies matching data hashes, sources, environment, training seed and split rows before comparing completed runs. `scripts/diagnose_mapper.py CHECKPOINT --output FILE.json` separately measures variation entering and leaving the mapper and the effect of changing the label at fixed noise. Its summaries locate attenuation; they do not establish the cause of collapse.

Tests verify that all three adversarial passes receive matching labels, generator initialization stays unchanged, quantum gradients remain nonzero, and historical discriminator dimensions remain compatible.

These experiments use analytic quantum simulation and backpropagation. The paper's finite-shot estimation and parameter-shift training remain documented differences. Hidden widths, learning rates, exact data preparation and downstream classifier settings are implementation assumptions. This is a paper-based architecture investigation, not a verified reproduction or hardware result.

## Completed results

All six 30-epoch runs completed. At validation-selected checkpoints, mean test macro F1 increased from **0.5184 to 0.5767**, AUROC from **0.5767 to 0.7239**, and average precision from **0.2035 to 0.3106**. Mean accuracy decreased from **0.7763 to 0.7407**. Macro F1 and AUROC improved in each paired seed, but the epoch-30 comparison did not show a consistent benefit. Both variants still show severe collapse, so these are early-checkpoint gains rather than a solved benchmark.

See the [full three-seed report, variability and diagnostic plots](results/conditioning_comparison/REPORT.md). All 25 unit tests passed. The next paper-based optimization work should retain the documented mapper activations and test the unspecified training settings using validation and diversity diagnostics.


## 100-epoch follow-up

Three new runs extended the same settings to 100 epochs, exactly reproducing the earlier first 30 loss epochs. They did not show sustained accuracy improvement. Mean final-epoch test accuracy was 56.8%, macro F1 0.4365 and AUROC 0.5986, with severe collapse. See the [full learning curves and diagnostics](results/conditioning_100_summary/REPORT.md). These runs reuse the earlier exploratory splits; they are not an independent confirmation.
