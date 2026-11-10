# Make quantum variation visible to the existing mapper

## Why this experiment follows the diagnostics

Frozen-model checks found that quantum probabilities vary with random input, but fixed-label generated rows barely vary. Discriminator real/fake accuracy approaches 99%. A learning-rate comparison did not fix this: slower D hurt average utility, and faster G achieved majority-class accuracy with AUROC 0.5 and essentially no diversity.

This experiment changes **only the initial coefficients of the mapper's first Linear layer**. The probability input, one-hot label, ReLU layers, linear output, quantum circuit and adversarial losses remain unchanged. The supplied paper defines that architecture in §3.1.3 but does not give a reproducible classical weight initialization. This gain is our experiment, not an author-confirmed paper setting. The existing final-H circuit readout remains an explicit paper extension.

## What is changed mathematically?

Write the first layer's original preactivation as:

`a(p, y) = Wq p + Wy y + b`

Estimate `mu`, the quantum probability mean, using 256 analytic circuit evaluations on random latent angles. The calibration uses **no training, validation or test rows**. A separate local random generator ensures that this does not shift the training RNG.

For gain `g`, initialize:

```text
Wq_new = g * Wq
b_new  = b + (1 - g) * Wq * mu
Wy_new = Wy
```

This gives:

`a_new(p, y) = g * Wq * (p - mu) + Wq * mu + Wy * y + b`

At `p = mu`, the old and new preactivation match. Variation around mu is amplified by g. Label weights, later mapper weights, discriminator weights and quantum angles keep their original initial values. Subsequent nonlinear layers mean final generated-row variation need not scale exactly by g.

This is folded into ordinary stored weights and bias. There is **no extra inference layer, feature normalization step, probability replacement, or new loss**. Checkpoints retain the same state-dictionary shapes. The optional `quantum_init_gain` defaults to 1, preserving historical initialization. Existing configuration files lacking it still load.

## Controlled experiment

Gain 1 is the control. Gain 16 (sqrt(256), a heuristic scale rather than a paper number) was tested first. It improved the mean minimum within-class standard-deviation ratio approximately 8-fold at selected checkpoints, but worsened test AUROC and numeric KS. A moderate gain 4 was then added as an exploratory follow-up. This is sequential screening on already examined splits, not independent confirmation.

Every gain uses seeds 42/43/44, 30 epochs, two quantum layers, the same classical seed, same train/validation/test rows, same latent/minibatch RNG, batch size 64, and both learning rates at 0.0002. Checkpoints use validation macro F1. Distribution and diversity diagnostics are required alongside predictive scores.

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper_readout --condition-discriminator --layers 2 \
  --seed 42 --classical-init-seed 42042 --epochs 30 --eval-every 10 --batch-size 64 \
  --lr-generator 0.0002 --lr-discriminator 0.0002 --quantum-init-gain 4 \
  --output results/initialization_next
```

Source runs are grouped in `results/quantum_initialization_study/runs/gain_G/seed_S/`. The original gain-1-versus-16 screen is in `analysis/`; the complete comparison is in `analysis_all_gains/`.

## Verification

Tests check preservation of the calibrated mean response, exact perturbation scaling in double precision, unchanged global RNG, unchanged quantum/discriminator/later-mapper starting weights, continued quantum gradients and checkpoint reloads. The model architecture and default loss were not replaced.

During reload verification, the hardware inference loader was found to infer feature width from the discriminator input. That is wrong for D(x,y), whose input also includes labels. It now derives feature width from the mapper output, and a regression test loads a conditional gained checkpoint. No IonQ job was submitted; export/inference compatibility tests are local.

A gain that increases spread but harms useful label relationships is not an improvement. No benchmark or hardware-readiness claim follows from this initialization alone.


## Completed result

All nine runs completed. Gain 4 improved validation-selected mean test accuracy **69.4% → 77.0%**, macro F1 **0.5453 → 0.6013**, and AUROC **0.7428 → 0.7923**. Raw within-class variation increased approximately 4.9× and numeric KS improved slightly. AP decreased, and fixed-epoch-30 macro F1 was worse; diversity remains far below the real data. Gain 16 harmed utility. Gain 4 is an experimental candidate only; defaults are unchanged.

[Full results and tradeoffs](results/history/quantum_initialization_study/analysis_all_gains/REPORT.md) · [Validation curves](results/history/quantum_initialization_study/analysis_all_gains/validation_curves.png) · [Diversity checks](results/history/quantum_initialization_study/analysis_all_gains/diversity_checks.png). All 32 tests passed and the default teaching notebook's results remained exactly unchanged.
