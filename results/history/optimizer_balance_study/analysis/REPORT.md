# Generator/discriminator learning-rate balance

Each candidate changes only one learning rate from the control. Same two-layer readout circuit, mapper, D(x,y), objective, split and starting weights within paired seeds. All runs completed 30 epochs.

The supplied paper leaves rates unspecified. This is tuning an implementation choice, not recovering an author-confirmed setting. No new objective or output activation is added. The existing final-H readout remains an explicit paper extension.

Checkpoints are selected by validation macro F1. Test results are reported after selection, not used for it. Means ± sample SD over the paired seeds; short reused-split runs are exploratory, not independent confirmation.

| Variant | G rate | D rate | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.0002 | 0.0002 | best | 0.5409 ± 0.0230 | 0.6937 ± 0.0369 | 0.5453 ± 0.0247 | 0.7428 ± 0.0026 | 0.3701 ± 0.0310 |
| control | 0.0002 | 0.0002 | last | 0.5028 ± 0.0540 | 0.6327 ± 0.0704 | 0.5057 ± 0.0440 | 0.7383 ± 0.0085 | 0.3571 ± 0.0351 |
| faster_g | 0.0008 | 0.0002 | best | 0.4919 ± 0.0296 | 0.8420 ± 0.1074 | 0.4965 ± 0.0377 | 0.5823 ± 0.1425 | 0.1815 ± 0.1480 |
| faster_g | 0.0008 | 0.0002 | last | 0.4748 ± 0.0000 | 0.9040 ± 0.0000 | 0.4748 ± 0.0000 | 0.5000 ± 0.0000 | 0.0960 ± 0.0000 |
| slower_d | 0.0002 | 5e-05 | best | 0.4409 ± 0.1900 | 0.5700 ± 0.3111 | 0.4430 ± 0.2116 | 0.5503 ± 0.2826 | 0.2113 ± 0.1382 |
| slower_d | 0.0002 | 5e-05 | last | 0.4409 ± 0.1900 | 0.5700 ± 0.3111 | 0.4430 ± 0.2116 | 0.5503 ± 0.2826 | 0.2113 ± 0.1382 |

[Validation curves](validation_curves.png) · [Loss curves](loss_curves.png) · [Diversity curves](diversity_curves.png) · [All per-seed scores](final_scores.csv)

A discriminator near perfect real/fake accuracy establishes an easy discrimination task, not a proof that a slower learning rate will fix collapse. Non-saturating generator BCE also does not imply its logit gradient vanishes when D is confident. The intervention must be judged empirically using both utility and diversity.

Original audit: `results/training_balance_audit/`. Source runs:
- `results/optimizer_balance_study/runs/control/seed_42`
- `results/optimizer_balance_study/runs/control/seed_43`
- `results/optimizer_balance_study/runs/control/seed_44`
- `results/optimizer_balance_study/runs/faster_g/seed_42`
- `results/optimizer_balance_study/runs/faster_g/seed_43`
- `results/optimizer_balance_study/runs/faster_g/seed_44`
- `results/optimizer_balance_study/runs/slower_d/seed_42`
- `results/optimizer_balance_study/runs/slower_d/seed_43`
- `results/optimizer_balance_study/runs/slower_d/seed_44`


## Outcome

Neither rate candidate improved average validation macro F1 over the control: **0.5409** control, **0.4409** slower D, **0.4919** faster G. The faster-G final classifiers achieved **90.4% accuracy with AUROC 0.5, AP 0.096 and zero minority-class recall**, exactly the majority-class outcome. Generated within-class variance essentially disappeared. This is a rejected collapse, not an accuracy improvement.

Selected-checkpoint test AUROC was **0.7428** for the control, **0.5503** for slower D and **0.5823** for faster G. The original hypothesis that simply changing the rate balance would solve collapse was not supported. Keep the existing default rates; these results do not establish that every possible rate schedule fails.

A separate initialization test was then motivated by the mapper-variation audit. It preserves the same network/loss and is recorded in `results/quantum_initialization_study/`; it is not silently combined with these learning-rate candidates.
