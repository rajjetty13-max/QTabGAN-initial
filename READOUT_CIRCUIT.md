# A targeted circuit change: make final phases measurable

The current circuit-only experiment keeps the paper's eight-qubit RY/RZ variational layers and ordered circular CNOT chain. Mapper, discriminator, preprocessing, learning rates, batch size and quantum/classical initial parameter values are held fixed for each paired seed.

## The change

Select `ansatz="paper_readout"` or `--ansatz paper_readout`. This adds a Hadamard on every wire immediately before measurement. Computational-basis measurement after those gates is equivalent to X-basis readout of the preceding state. It can convert the final RZ phase changes into probability changes.

This is an explicit extension to the paper's circuit, not a gate shown in Figure 3 or an exact reproduction claim. The internal paper blocks, parameter order, entanglers and 256-dimensional probability output are retained. The alternative interpretation that repeats Hadamards between layers is separately selectable with `--hadamards-per-layer`. Both options can be combined, and the experiment screens them separately and together.

The existing `reuploading` circuit is retained for historical comparison; it is not used by this new readout experiment. The default remains `paper`, preserving historical checkpoints. Do not change an old checkpoint's ansatz and call it a newly trained model: the probabilities supplied to its trained mapper would change.

## Measured circuit properties

Across seeds 42, 43 and 44, with two layers and 32 latent samples per seed:

| Property | Original | Original + final Hadamards |
| --- | ---: | ---: |
| Trainable parameters | 32 | 32 |
| Parameters affecting probabilities | 24 | 32 |
| Stacked probability-Jacobian rank | 24 | 32 |
| CNOT gates | 16 | 16 |
| Total gates | 64 | 72 |
| Logical depth | 22 | 23 |

This fixes invisible final phase parameters, not circuit depth. The added readout gates may increase physical error; no hardware noise advantage is claimed. Rank is measured over stacked latent samples, and nonzero parameter sensitivity alone does not prove good GAN optimization.

The mean pairwise total-variation distance between probability vectors decreased in these untrained checks, while entropy increased. That is a tradeoff, not proof of better diversity. The generated-row tests below are needed to judge utility.

[Structural metrics](results/readout_circuit_diagnostics/summary.csv) · [Parameter sensitivity plot](results/readout_circuit_diagnostics/parameter_sensitivity.png) · [Original diagram](results/readout_circuit_diagnostics/paper_seed_42/circuit.png) · [New diagram](results/readout_circuit_diagnostics/paper_readout_seed_42/circuit.png)

Tests verify that the only added operations are final Hadamards, all parameter gradients can be nonzero, backprop agrees with parameter-shift, finite-shot sampling remains normalized, and exported IonQ QIS reproduces local probabilities. Training, shot sampling and export all use the same operation definition. No cloud job was submitted.

## Training experiment

The seed-42 screen compares four circuits for 30 epochs: original, repeated layer Hadamards, final readout Hadamards, and both changes. Validation is measured at epochs 1/10/20/30. A further paired check compares original versus final readout on seeds 43 and 44. The paper-supported D(x,y) discriminator is used throughout.

The simple readout change was taken forward because it fixes the identified inactive phases and was essentially tied with the original's best validation macro F1 in the seed-42 screen. This is exploratory candidate selection, not independent confirmation. Checkpoints use validation macro F1, not test scores. Previously inspected Loan splits are reused, so test metrics remain exploratory.

Example for a new run:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python extended_benchmark.py \
  --data data/loan_prepared.csv --target PersonalLoan \
  --categorical 'ZIP Code' Education SecuritiesAccount CDAccount Online CreditCard \
  --ansatz paper_readout --condition-discriminator --seed 42 \
  --epochs 30 --eval-every 10 --batch-size 64 \
  --output results/readout_next
```

For the original control, use `--ansatz paper`. For the paper's repeated-H interpretation, add `--hadamards-per-layer`. No extra loss, output activation or classical network change is introduced.


## Completed training results

Eight 30-epoch runs completed: four initial variants at seed 42 and paired original/readout checks at seeds 43 and 44. At the fixed epoch-30 endpoint, mean accuracy improved **53.7% → 62.4%**, macro F1 **0.4425 → 0.5039**, and AUROC **0.7137 → 0.7381**. With validation-selected checkpoints, AUROC improved but accuracy and macro F1 decreased. Both variants remain collapsed; this is a mixed result, not a solved benchmark. The default circuit is unchanged.

[Full experiment report](results/circuit_readout_screen/REPORT.md) · [Paired score plots](results/circuit_readout_screen/paired_test_scores.png) · [Validation curves](results/circuit_readout_screen/validation_curves.png)


## Follow-up: quantum layer count

The [one/two/three-layer study](results/readout_depth_study/analysis/REPORT.md) held classical initialization fixed and tested three seeds per depth. Two layers retained the best validation macro F1; one layer halved CNOTs and improved epoch-30 AUROC with an accuracy/F1 tradeoff. All nine source runs are grouped in `results/readout_depth_study/`.
