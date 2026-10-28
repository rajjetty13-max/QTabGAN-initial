# Matched CTGAN versus QTabGAN

Seed 42; identical raw features, train/validation/test row indices, train-fitted evaluation encoder and random-forest evaluator. Both GANs trained for 100 epochs. This is equal epochs, not equal compute or identical GAN internals. CTGAN uses its own internal preprocessing.

| Model | Test accuracy | Macro F1 | AUROC | Average precision |
| --- | ---: | ---: | ---: | ---: |
| QTabGAN epoch 100 | 0.4740 | 0.3652 | 0.4103 | 0.0866 |
| QTabGAN selected epoch 10 | 0.5810 | 0.4787 | 0.6696 | 0.1932 |
| SDV CTGAN epoch 100 | 0.8990 | 0.6245 | 0.8711 | 0.4091 |
| Real-trained reference | 0.9830 | 0.9473 | 0.9966 | 0.9778 |

The two final-epoch rows are the direct duration-matched comparison. QTabGAN’s selected checkpoint is additional context, chosen by validation macro F1; CTGAN was evaluated only at its final epoch. This single-seed comparison must not be presented as a multi-seed model ranking. SDV CTGAN differs from the paper’s CTAB-GAN and CTAB-GAN+ baselines.

[ROC and precision–recall comparison](auc_comparison.png) · [Numbers and confusion counts](summary.csv)

Source QTabGAN folder: `results/conditioning_trial_100_seed_42`. Source CTGAN folder: `results/ctgan_matched_100_seed_42`. Each has loss curves and predictive diagnostics. CTGAN additionally retains raw numeric distribution and class-specific diversity diagnostics. Its model is the SDV implementation imported from the installed ctgan package, not a second implementation in qtabgan/.
