# Current results

The active work compares the improved QTabGAN generator at 30 and 100 epochs. The same three seeds and data splits are retained; checkpoints use validation macro F1.

| Folder | What to open |
| --- | --- |
| [100-epoch continuation](tabular_moment_100_study/analysis/REPORT.md) | Three continued runs from epoch 30; 91.0% mean selected-checkpoint accuracy, 0.7538 macro F1 |
| [30-epoch model comparison](tabular_moment_study/analysis/REPORT.md) | Control, valid categorical output, and output plus class-conditional moments; nine matched runs |
| [Threshold calibration](tabular_moment_study/threshold_calibration/REPORT.md) | Separate validation-only classifier calibration and its recall tradeoff |
| [CTGAN baseline](history/conditioning_100_summary/ctgan_comparison/REPORT.md) | Matched seed-42 SDV CTGAN comparison; raw artifacts remain in `ctgan_matched_100_seed_42/` |
| [Paper reference](paper_reference.json) | Published values and unresolved reproduction settings |
| [Historical summaries](history/) | Compact circuit, conditioning, initialization, optimizer and depth comparisons |

The 100-epoch continuation averaged **91.0% accuracy, 0.7538 macro F1 and 0.8932 AUROC** at validation-selected checkpoints (epochs 80/90/70). The actual final epoch averaged 89.7% accuracy and 0.7042 macro F1. Minority recall decreased from 71.5% to 59.7%; these are exploratory reused-split results.

The 30-epoch moment candidate averaged **85.2% accuracy, 0.6971 macro F1 and 0.8764 AUROC** at validation-selected checkpoints. The often-quoted **87.2%** was seed 43, not the three-seed mean. [Model changes and reproduction command](../TABULAR_OUTPUT.md).

Each active seed folder retains `best.pt`, `latest.pt`, `manifest.json`, `split.npz`, `preprocessor.joblib`, complete loss/validation histories, and selected/final generated data and diagnostics. Model/data binaries are retained locally and may be gitignored. The `source/` folders preserve the exact executed code.

Older raw experiments were moved to `../archives/legacy_results_2026-11-10/results/`. They remain available for audit or restoration. [Relocation manifest and size accounting](history/relocation_manifest.json) records every move. Existing summary links were updated; no unique experimental evidence was deleted.
