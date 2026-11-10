# Readout circuit: 100-epoch continuation

**Complete:** [Comparison report](analysis/REPORT.md) · [Validation curves](analysis/validation_curves.png) · [Diversity curves](analysis/diversity_curves.png). All six runs reached epoch 100.

This study extends copies of the completed one- and two-layer 30-epoch runs to epoch 100, for seeds 42/43/44. The original study in `../readout_depth_study/` is preserved.

- `runs/layers_L/seed_S/`: each resumed run, with its complete epoch history and checkpoints.
- `epoch_30_report.json` inside each run: the copied 30-epoch final report.
- `continuation.json`: source run, starting checkpoint SHA256, and requested final epoch.

Both depths retain the original paper RY/RZ blocks and circular CNOTs, experimental final-H readout, and D(x,y). Each paired seed has identical classical initialization, data rows, training RNG and optimization settings. Checkpoints are selected by validation macro F1. Equal epochs do not mean equal gate count or compute.

This is an exploratory extension on previously viewed splits. Its test scores are not independent confirmation. While training is active, inherited `test/` files still describe epoch 30; a new `final_report.json` appears only once the 100-epoch run and its final evaluation finish.


The early one-layer AUROC gain was transient. Best validation-selected mean test macro F1 was 0.5536 for one layer and 0.5603 for two; both remained collapsed. Use each run's `best.pt` for inference experiments and preserve `latest.pt` for continuation. The one-layer variant retains half the CNOTs, but is not a solved benchmark.
