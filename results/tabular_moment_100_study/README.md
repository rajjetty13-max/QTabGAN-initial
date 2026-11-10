# QTabGAN through 100 epochs

All three seeds completed 100 total epochs by continuing the retained epoch-30 checkpoints, including Adam and RNG states. [Full results and continuation audit](analysis/REPORT.md).

| Mean selected-checkpoint test metric | 30 epochs | 100 epochs |
| --- | ---: | ---: |
| Accuracy | 85.20% | 90.97% |
| Macro F1 | 0.6971 | 0.7538 |
| AUROC | 0.8764 | 0.8932 |
| AP | 0.4093 | 0.5229 |
| Minority recall | 71.53% | 59.72% |

Checkpoints were selected by validation macro F1 at epochs 80/90/70, with test accuracies 90.9%/91.3%/90.7%. The actual epoch-100 endpoints averaged 89.73% accuracy and 0.7042 macro F1. Keep `runs/moment_matched/seed_S/best.pt` for inference and `latest.pt` for exact continuation. Final checkpoints preserve full optimizer and RNG states.

The model uses bounded numeric output, straight-through categorical samples, moment weight 10, initialization gain 4, two-layer final-H readout and D(x,y). These remain explicit extensions to the paper; reused splits make these results exploratory. Higher accuracy/F1 came with lower minority recall. No new threshold calibration was applied.

`source/` retains the exact executed source. `analysis/continuation_audit.csv` verifies unchanged epoch-1–30 histories, row splits, preprocessing and source hashes, and Adam steps increasing from 1,410 to 4,700. The original 30-epoch comparison remains in `../tabular_moment_study/`.
