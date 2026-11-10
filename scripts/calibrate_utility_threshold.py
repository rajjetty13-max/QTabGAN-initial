"""Calibrate decision thresholds on validation predictions of preselected checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, recall_score, roc_auc_score)


VARIANTS = ('control', 'typed_output', 'moment_matched')
SEEDS = (42, 43, 44)
METRICS = ('accuracy', 'f1_macro', 'minority_recall', 'roc_auc', 'average_precision')


def read(path):
    return json.loads(path.read_text())


def score(truth, probability, threshold, minority):
    predicted = (probability > threshold).astype(int)
    return {'accuracy': float(accuracy_score(truth, predicted)),
            'f1_macro': float(f1_score(truth, predicted, labels=[0, 1], average='macro', zero_division=0)),
            'minority_recall': float(recall_score(truth, predicted, pos_label=minority, zero_division=0)),
            'roc_auc': float(roc_auc_score(truth, probability)),
            'average_precision': float(average_precision_score(truth, probability)),
            'confusion_matrix': confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, default=Path('results/tabular_moment_study'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = args.output or args.study / 'threshold_calibration'
    if out.exists():
        parser.error('Choose a fresh calibration folder; existing results will not be overwritten')
    thresholds = np.linspace(0, 1, 101)
    rows = []; selection = []; grid_rows = []; input_hashes = []; paired = {}; predictions = []
    for variant in VARIANTS:
        for seed in SEEDS:
            root = args.study / 'runs' / variant / f'seed_{seed}'
            report_file = root / 'final_report.json'
            if not report_file.exists():
                parser.error(f'A completed run is required: {root}')
            report = read(report_file); epoch = int(report['best_epoch'])
            validation_scores = pd.read_csv(root / 'validation_scores.csv')
            original_best = int(validation_scores.loc[validation_scores.f1_macro.idxmax(), 'epoch'])
            if original_best != epoch:
                parser.error(f'Original checkpoint does not follow its validation selection rule: {root}')
            counts = pd.read_csv(root / 'inputs' / 'class_counts.csv').set_index('class_index')['count']
            minority = int(counts.idxmin())
            folders = {'validation': root / 'validation' / f'epoch_{epoch}',
                       'test': root / 'test' / 'best'}
            frames = {}
            with np.load(root / 'split.npz') as stored:
                split = {key: stored[key].copy() for key in ('train', 'validation', 'test')}
            for dataset, folder in folders.items():
                path = folder / 'predictions.csv'
                frame = pd.read_csv(path)
                needed = {'truth', 'prediction', 'positive_probability'}
                if not needed.issubset(frame.columns) or len(frame) != len(split[dataset]):
                    parser.error(f'Invalid predictions file: {path}')
                probability = frame.positive_probability.to_numpy()
                if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
                        or set(frame.truth.unique()) != {0, 1}):
                    parser.error(f'Invalid binary truth/probabilities: {path}')
                if not np.array_equal((probability > .5).astype(int), frame.prediction.to_numpy()):
                    parser.error(f'>0.5 does not exactly reproduce saved predictions: {path}')
                frames[dataset] = frame
                input_hashes.append(dict(variant=variant, seed=seed, dataset=dataset,
                                         source_file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if seed in paired:
                previous = paired[seed]
                if (any(not np.array_equal(split[key], previous['split'][key]) for key in split)
                        or any(not np.array_equal(frames[key].truth.to_numpy(), previous['truth'][key])
                               for key in frames) or minority != previous['minority']):
                    parser.error(f'Paired splits, truth or minority class differ at seed {seed}')
            else:
                paired[seed] = dict(split=split, truth={key: frame.truth.to_numpy() for key, frame in frames.items()},
                                    minority=minority)

            validation = frames['validation']
            # Threshold selection uses only validation truth/probabilities.
            grid_scores = [float(f1_score(validation.truth,
                                          validation.positive_probability.to_numpy() > threshold,
                                          labels=[0, 1], average='macro', zero_division=0))
                           for threshold in thresholds]
            maximum = max(grid_scores)
            tied = [threshold for threshold, value in zip(thresholds, grid_scores) if value == maximum]
            selected = min(tied, key=lambda threshold: (round(abs(threshold - .5), 12), -threshold))
            grid_rows.extend(dict(variant=variant, seed=seed, epoch=epoch, threshold=float(threshold),
                                  validation_macro_f1=value, selected=bool(threshold == selected))
                             for threshold, value in zip(thresholds, grid_scores))
            selection.append(dict(variant=variant, seed=seed, epoch=epoch, threshold=float(selected),
                                  validation_macro_f1_original=float(f1_score(validation.truth, validation.prediction,
                                                                            labels=[0, 1], average='macro', zero_division=0)),
                                  validation_macro_f1_calibrated=maximum, tied_maxima=len(tied)))
            for dataset, frame in frames.items():
                for decision_rule, threshold in (('original', .5), ('calibrated', selected)):
                    metrics = score(frame.truth, frame.positive_probability, threshold, minority)
                    if dataset == 'test' and decision_rule == 'original':
                        for key in ('accuracy', 'f1_macro', 'roc_auc', 'average_precision'):
                            if not np.isclose(metrics[key], report['validation_selected'][key], rtol=0, atol=1e-12):
                                parser.error(f'Saved test scores and prediction-derived {key} differ: {root}')
                    rows.append(dict(variant=variant, seed=seed, checkpoint='original_validation_best', epoch=epoch,
                                     dataset=dataset, decision_rule=decision_rule, threshold=float(threshold),
                                     **{key: metrics[key] for key in METRICS},
                                     confusion_matrix=json.dumps(metrics['confusion_matrix'])))
                predictions.append(frame.assign(variant=variant, seed=seed, dataset=dataset,
                                                epoch=epoch, calibrated_threshold=float(selected),
                                                calibrated_prediction=(frame.positive_probability > selected).astype(int)))
    final = pd.DataFrame(rows); selected = pd.DataFrame(selection)
    # AUROC and AP depend on the unchanged ranking probabilities, not thresholds.
    for (_, _, _), part in final.groupby(['variant', 'seed', 'dataset']):
        for metric in ('roc_auc', 'average_precision'):
            if part[metric].nunique() != 1:
                parser.error(f'{metric} changed under threshold calibration')
    test = final[final.dataset == 'test']
    aggregate = test.groupby(['variant', 'decision_rule'])[list(METRICS)].agg(['mean', 'std'])
    out.mkdir(parents=True)
    for name, frame in {'scores': final, 'selected_thresholds': selected, 'threshold_grid': pd.DataFrame(grid_rows),
                        'predictions': pd.concat(predictions, ignore_index=True),
                        'input_hashes': pd.DataFrame(input_hashes)}.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    aggregate.to_csv(out / 'aggregate_test.csv')

    def cell(part, metric):
        return f'{part[metric].mean():.4f} ± {part[metric].std():.4f}'

    lines = ['# Validation-only decision-threshold calibration', '',
             'Every model keeps the checkpoint already selected by its original validation macro F1. '
             'The generator candidate remains **moment_matched**, chosen in the original validation-only model comparison. '
             'Calibration does not reselect checkpoints or rank model variants.', '',
             'For each group and seed, the decision threshold maximizes macro F1 using only predictions from '
             'that checkpoint’s validation epoch. The grid is `np.linspace(0, 1, 101)`. Ties choose the threshold '
             'closest to 0.5, then the larger threshold. A row is positive only when `positive_probability > threshold`; '
             'this preserves the random forest’s negative decision on a 0.5 tie.', '',
             'Each threshold was applied once to the saved predictions for that checkpoint’s test set. '
             'No test score, label or probability was used to choose thresholds or candidate models. '
             'All validation/test saved labels were verified to equal the original `probability > 0.5` rule, '
             'paired split rows and truth were checked, and original test metrics were reproduced from predictions.', '',
             '## Test results', '',
             'Means ± sample SD across seeds 42, 43 and 44. Original means use threshold 0.5. '
             'AUROC and AP remain exactly unchanged because thresholding does not change the probabilities.', '',
             '| Variant | Decision rule | Accuracy | Macro F1 | Minority recall | AUROC | AP |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for variant in VARIANTS:
        for rule in ('original', 'calibrated'):
            part = test[(test.variant == variant) & (test.decision_rule == rule)]
            lines.append('| ' + ' | '.join([variant, rule, *[cell(part, metric) for metric in METRICS]]) + ' |')
    lines += ['', '## Frozen checkpoints and thresholds', '',
              '| Variant | Seed | Original best epoch | Threshold | Validation macro F1 before | Validation macro F1 after |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in selected.itertuples(index=False):
        lines.append(f'| {row.variant} | {row.seed} | {row.epoch} | {row.threshold:.2f} | '
                     f'{row.validation_macro_f1_original:.4f} | {row.validation_macro_f1_calibrated:.4f} |')
    lines += ['', '## Scope', '',
              'This is a separate evaluation extension, not an improvement to generator training, data fidelity '
              'or quantum circuitry. Calibration can improve an older model too; all three groups receive '
              'the same procedure. The moment objective and typed output remain distinct model extensions '
              'to the paper. No model was retrained, and no raw report or prediction file was overwritten.', '',
              'These validation/test splits have been examined during earlier exploratory work. '
              'This additional evaluation is exploratory and does not establish independent confirmation '
              'or statistical significance. Macro-F1 calibration may trade accuracy against minority recall. '
              'The 90.4% majority-class accuracy reference remains relevant.', '',
              '[Per-seed validation and test scores](scores.csv) · [Selected thresholds](selected_thresholds.csv) · '
              '[Full validation grid](threshold_grid.csv) · [Calibrated predictions](predictions.csv) · '
              '[Input hashes](input_hashes.csv)']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(aggregate.to_string())
    print(f'Calibration report: {out / "REPORT.md"}')


if __name__ == '__main__':
    main()
