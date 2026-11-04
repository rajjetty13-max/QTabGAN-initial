# Improve the model by checking diagnostics against the paper

## Review before intervention

The previous depth/readout trials had finite quantum gradients but severe collapse. More layers or epochs did not reliably improve useful data. The teaching run also showed almost constant values within each class and one generated category per categorical feature. That tiny teaching run is not the benchmark; the frozen-model audit below uses full-data study checkpoints.

`scripts/audit_training_balance.py` inspected the two-layer 100-epoch models at seeds 42/43/44, using training and validation rows only. At their final checkpoints, mean D(real validation) was approximately **0.983**, D(generated) **0.022**, and balanced train real/fake accuracy **0.992**. The discriminator's distinction generalizes beyond its training rows; it is not merely a high training score. Quantum probability variation remains nonzero, while generated fixed-label variation is very small.

[Audit numbers](results/training_balance_audit/audit.csv) · [Discriminator confidence plot](results/training_balance_audit/discriminator_confidence.png)

This establishes an easy discrimination task and weak generated diversity. It does not prove a single cause. In particular, the non-saturating generator BCE does not automatically have a vanishing derivative with respect to its logit just because D is confident. RMS values in probability space and encoded feature space have different units and cannot be interpreted as an information-retention percentage.

## What the paper says

The supplied `2602.12704v2.pdf`, §§3.1.3–3.2, defines ReLU feedforward mapper/discriminator hidden layers, a linear mapper output and a scalar D(x,y). §4.2 describes alternating adversarial updates and gradient optimization but does not give a complete reproducible learning-rate configuration.

We therefore retain the model and losses and test learning rates as **implementation choices**, not author-confirmed numbers. No output sigmoid/Gumbel layer, auxiliary classifier, feature-matching objective or diversity penalty is added. The baseline circuit is the current two-layer paper-based circuit with final-H readout, which remains an explicit extension to the paper. All three variants use the same extension; it is not the factor under test.

## Controlled single-factor study

| Variant | Generator/mapper rate | Discriminator rate | Difference from control |
| --- | ---: | ---: | --- |
| control | 0.0002 | 0.0002 | None |
| slower_d | 0.0002 | 0.00005 | D four times slower |
| faster_g | 0.0008 | 0.0002 | G four times faster |

Each variant runs 30 epochs across seeds 42/43/44. The faster-G candidate was added after early validation showed the slower-D candidate lagging. This is exploratory sequential screening, not preregistered confirmation. Within each seed, the split, class conditions, quantum initial angles, classical initial weights, training RNG, batch size, architecture, update ratio and evaluator remain fixed. Only one rate changes in each candidate.

Checkpoints use validation macro F1, with diversity as a required diagnostic. Final test reporting occurs after training and selection. These are previously viewed Loan splits; test scores are not independent confirmation. Equal epochs are not an equal-compute comparison with CTGAN.

Run a new variant in a fresh output folder:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper_readout --condition-discriminator --layers 2 \
  --seed 42 --classical-init-seed 42042 --epochs 30 --eval-every 10 \
  --batch-size 64 --lr-generator 0.0008 --lr-discriminator 0.0002 \
  --output results/rate_trial_next
```

The new CLI flags expose existing `QTabGANConfig` fields. Defaults remain 0.0002 for both rates and old checkpoint configurations still load. New manifests record both rates; strict resume checks reject a protocol change rather than silently changing a continuing run. Previous studies were not overwritten.

## Evidence to keep

All new runs are grouped in `results/optimizer_balance_study/runs/VARIANT/seed_S/`. The analysis verifies data/source/environment consistency, exact paired split indices, initialization seeds, epoch budgets and that every candidate changes at most one rate. It saves validation, loss and diversity plots and both selected/final test results.

A lower discriminator loss is not the selection target. An intervention that makes losses look balanced while harming validation utility or retaining collapsed data is not an improvement. No hardware or gate-noise simulation is part of this study.


## Completed outcomes and next candidate

All nine learning-rate trials completed and neither rate candidate improved average validation macro F1. Faster G's apparent 90.4% accuracy was a majority-class collapse (AUROC 0.5). [Rate comparison](results/optimizer_balance_study/analysis/REPORT.md).

The follow-up tested a mapper-initialization change without changing architecture or loss: [derivation](QUANTUM_INITIALIZATION.md). Of gains 1/4/16, gain 4 improved selected-checkpoint test accuracy, macro F1 and AUROC across the three-seed mean, with about 5× more numeric within-class variation. It remains collapsed and unstable; AP decreased. [Full evidence](results/quantum_initialization_study/analysis_all_gains/REPORT.md).

This was 18 short controlled runs, not a successful paper reproduction. Defaults remain the original rates and gain 1. Gain 4 can be selected explicitly with `--quantum-init-gain 4`. The next evaluation should extend this matched comparison without silently changing its circuit, loss, split or selection criterion.
