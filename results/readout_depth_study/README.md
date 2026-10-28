# Readout circuit depth study

Start with [the comparison report](analysis/REPORT.md). It covers nine completed 30-epoch runs: 1/2/3 layers across seeds 42/43/44.

- `analysis/`: validation, loss and diversity curves; aggregate and per-seed scores; circuit diagrams for each depth.
- `runs/layers_L/seed_S/`: complete training artifacts, manifests, checkpoints, synthetic data and predictive diagnostics.

Two layers won the declared validation macro-F1 comparison. One layer reduced CNOTs from 16 to 8 and improved epoch-30 AUROC, at the cost of accuracy and macro F1. Both remain exploratory candidates with severe collapse. See [the experiment protocol](../../DEPTH_EXPERIMENT.md).
