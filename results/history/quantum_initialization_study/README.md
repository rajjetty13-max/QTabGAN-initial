# Quantum-aware mapper initialization

Start with the [complete comparison](analysis_all_gains/REPORT.md). It includes nine completed 30-epoch runs across gains 1/4/16 and seeds 42/43/44.

- `runs/gain_G/seed_S/`: checkpoints, losses, generated rows, manifests and validation/test diagnostics.
- `analysis_all_gains/`: all three gains, including validation/utility and diversity checks.
- `analysis/`: the earlier gain-1-versus-16 screen, retained as history.
- `audits/`: frozen-model confidence and mapper-sensitivity checks for gains 1 and 4.

Gain 4 improves selected-checkpoint mean accuracy/F1/AUROC and within-class spread. AP and final-epoch stability remain weaker, and collapse is not solved. It is opt-in, not a new default or reproduced paper setting.

See [the initialization derivation](../../../QUANTUM_INITIALIZATION.md) and [the diagnostic/paper review](../../../MODEL_BALANCE_REVIEW.md).
