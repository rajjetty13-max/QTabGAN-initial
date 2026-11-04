# Learning-rate balance study

[Full report](analysis/REPORT.md) · [Validation curves](analysis/validation_curves.png) · [Loss curves](analysis/loss_curves.png) · [Diversity curves](analysis/diversity_curves.png)

Nine completed 30-epoch runs compare control rates, slower discriminator, and faster generator across three seeds. Each candidate changes one rate only. Neither improves the control's average validation macro F1. Faster G's 90.4% endpoint accuracy is a majority-class collapse with AUROC 0.5.

All original settings remain defaults. Run artifacts are under `runs/VARIANT/seed_S/`. See the [paper and diagnostic review](../../MODEL_BALANCE_REVIEW.md).
