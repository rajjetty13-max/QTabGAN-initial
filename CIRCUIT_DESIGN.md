# A circuit worth testing before another long training run

The 1,000-epoch experiment was stopped at epoch 227 when we switched to circuit design. Its saved validation checkpoints at epochs 100 and 200 still show severe collapse. No final test result was produced. The interrupted run and its epoch-220 checkpoint were subsequently removed during cleanup; the completed 1–30 epoch comparisons below were retained.

## What was wrong with the original circuit?

The original circuit initializes each qubit with H and an input RY rotation. Each trainable layer then applies RY, RZ and an ordered CNOT ring.

In the last layer, RZ changes the phases of amplitudes. The remaining CNOT gates only permute computational-basis states, and measuring probabilities discards those phases. Consequently, the final eight RZ angles cannot change the generator's probability output. Updating those angles cannot help the classical mapper.

This is a structural limitation, not a complete explanation for GAN collapse. The discriminator does not see labels, and the mapper's unconstrained outputs and training dynamics can also cause problems.

## The experimental alternative

Select `ansatz="reuploading"` in `QTabGANConfig`. The default `ansatz="paper"` preserves the existing circuit and old checkpoint interpretation.

The alternative starts with H and repeats this sequence for each layer:

1. Encode the **same random input angles** again with RY.
2. Apply trainable RZ followed by trainable RY. The latter can turn phase differences into measurable probability differences.
3. Entangle using two staggered sets of disjoint CNOT pairs. Reverse the control direction on alternate layers.

This rearranges entanglement and changes the function being learned. It is an experimental extension, not a reproduction of the paper's circuit. It supports even qubit counts and retains the same weight tensor dimensions. An old checkpoint's angles are not trained for this alternative; do not switch its ansatz and treat it as a trained candidate model.

## What we actually measured

The diagnostic used eight qubits, two layers, seeds 42/43/44, and 32 random input vectors per seed. Both variants received identical angles and initial weights for a given seed. No GAN training or hardware calls were performed.

| Measurement | Original | Experimental |
| --- | ---: | ---: |
| Trainable angle slots | 32 | 32 |
| Angles affecting measured probabilities | 24 | 32 |
| Stacked probability-Jacobian rank | 24 | 32 |
| Logical gate depth | 22 | 11 |
| CNOT count | 16 | 16 |
| Total gates | 64 | 72 |
| Mean normalized probability entropy, three seeds | 0.8052 | 0.8426 |
| Mean pairwise total-variation distance, three seeds | 0.6406 | 0.6340 |

The first six rows were identical across the three seeds. Derivatives were computed using exact parameter shifts in double precision; an angle is counted as active if its RMS probability derivative exceeds 1e-9. The Jacobian stacks responses over all sampled inputs; its rank does not describe a single input's state alone.

The design removes the identified inactive parameters and halves the logical depth, at the cost of eight extra one-qubit gates. It has **not** improved every diversity measure: pairwise probability separation is slightly lower on average. Higher entropy is also not proof of more useful generated bank records.

Logical depth is the number of sequential gate layers inferred from wire dependencies. It is not IonQ-native depth, physical runtime, or a measured noise advantage. The same sixteen CNOTs remain, and hardware compilation may change the comparison.

## Open the evidence

- [Original circuit](results/circuit_comparison/paper_seed_42/circuit.png)
- [Experimental circuit](results/circuit_comparison/reuploading_seed_42/circuit.png)
- [Resource and sensitivity comparison](results/circuit_comparison/comparison.png)
- [Every parameter's probability sensitivity](results/circuit_comparison/parameter_sensitivity.png)
- [All six measured rows](results/circuit_comparison/summary.csv)
- [Reproducibility manifest](results/circuit_comparison/manifest.json)

Each seed's subfolder includes the exact circuit text and image, per-angle derivatives, Jacobian singular values, and metrics. Re-run diagnostics into a fresh directory:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python scripts/compare_circuits.py \
  --output results/circuit_comparison_next
```

The existing IonQ exporter supports all gates used here. Replaying exported QIS locally matched circuit probabilities to within 6.3e-17 across these checks; this is export correctness, not execution on an IonQ device. Tests also compare backpropagation with parameter-shift gradients and verify the new circuit's batched probabilities.

## How to use it when we return to training

```python
from qtabgan import QTabGANConfig
config = QTabGANConfig(ansatz="reuploading", n_qubits=8, n_layers=2)
```

Both `extended_benchmark.py` and the older `benchmark.py` accept `--ansatz reuploading`. Leave the flag out for the original circuit. Use separate output folders and train new weights for each variant. Keep data splits, seed, mapper, discriminator, learning rates, batch size and evaluation schedule identical when comparing them.

A circuit-only diagnostic cannot establish better GAN accuracy or reproduction of the paper's benchmark. The first matched training comparison is recorded below.

## First training comparison: completed

A matched 30-epoch full-Loan test used seed 42, batch size 64, the same stratified 60/20/20 split and the same classical networks. Checkpoints were selected by validation macro F1. On the final test set, the original circuit scored AUROC **0.7413**, macro F1 **0.5052**, and average precision **0.3536**; the alternative scored **0.7377**, **0.4592**, and **0.2895**. Both showed severe collapse. The selected epochs were 30 and 20 respectively.

The structural improvement did not yield better predictive utility in this short single-seed test. See the [complete report and plots](results/circuit_trial_30_comparison/REPORT.md). Keep the original as the performance baseline while investigating the mapper and conditioning path. More seeds are needed before claiming a reliable model ranking.


## Follow-up: retain the paper blocks and change readout

The new [final-H readout experiment](READOUT_CIRCUIT.md) keeps the original RY/RZ and ordered CNOT ring, rather than using reuploading. It makes all 32 angles effective with one added logical layer. Eight training runs tested it and the repeated-H interpretation; [the full report](results/circuit_readout_screen/REPORT.md) records fixed-budget gains, selected-checkpoint tradeoffs and remaining collapse.
