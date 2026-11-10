"""Verify and summarize exact 30-to-100 epoch conditional-moment continuations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


METRICS = ('accuracy', 'f1_macro', 'roc_auc', 'average_precision', 'minority_recall')
DIVERSITY = ('within_class_std_ratio', 'category_coverage')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recall(utility, minority):
    matrix = np.asarray(utility['confusion_matrix'])
    total = matrix[minority].sum()
    return float(matrix[minority, minority] / total) if total else 0.0


def diversity(folder):
    numeric = pd.read_csv(folder / 'raw_numeric_diagnostics.csv')
    category = pd.read_csv(folder / 'category_coverage.csv')
    return {'within_class_std_ratio': float(numeric.min_within_class_std_ratio.mean()),
            'category_coverage': float(category.training_category_coverage.mean())}


def optimizer_steps(state, key):
    return {float(item['step']) for item in state[key]['state'].values()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, default=Path('results/tabular_moment_100_study'))
    parser.add_argument('--previous', type=Path, default=Path('results/tabular_moment_study'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output folder')
    previous = args.previous / 'runs' / 'moment_matched'
    current = args.study / 'runs' / 'moment_matched'
    roots = sorted(current.glob('seed_*'))
    if {root.name for root in roots} != {'seed_42', 'seed_43', 'seed_44'}:
        parser.error('Require seeds 42, 43 and 44')
    if any(not (root / 'final_report.json').is_file() for root in roots):
        parser.error('All three 100-epoch runs must finish before summarizing')

    rows = []; curves = []; losses = []; variations = []; audit = []; source_audit = []
    reference = None
    per_seed_manifests = ('seed', 'classical_init_seed', 'training_seed')
    for root in roots:
        old = previous / root.name
        manifest = read(root / 'manifest.json')
        if manifest != read(old / 'manifest.json'):
            parser.error(f'Continuation changed source, environment or protocol: {root}')
        common = {key: value for key, value in manifest.items() if key not in per_seed_manifests}
        if reference is None:
            reference = common
        elif common != reference:
            parser.error(f'Unmatched settings across seeds: {root}')
        seed = manifest['seed']
        if root.name != f'seed_{seed}' or not manifest['tabular_output'] or manifest['moment_loss_weight'] != 10:
            parser.error(f'Unexpected seed or candidate settings: {root}')
        for name, expected in manifest['source_sha256'].items():
            actual = digest(args.study / 'source' / name)
            if actual != expected:
                parser.error(f'Launch snapshot does not match manifest: {name}')
            if root == roots[0]:
                source_audit.append({'source_file': name, 'recorded_sha256': expected,
                                     'launch_snapshot_sha256': actual})
        provenance = read(root / 'continuation.json')
        start = provenance.get('start_epoch', provenance.get('from_epoch'))
        target = provenance.get('target_epoch', provenance.get('requested_total_epochs'))
        recorded_digest = provenance.get('source_checkpoint_sha256', provenance.get('checkpoint_sha256'))
        if start != 30 or target != 100 or recorded_digest != digest(old / 'latest.pt'):
            parser.error(f'Continuation provenance does not identify the retained epoch-30 checkpoint: {root}')
        if (not provenance.get('optimizer_state_resumed') or not provenance.get('rng_state_resumed')):
            parser.error(f'Continuation must record resumed optimizer and RNG states: {root}')
        if Path(provenance['source_run']).resolve() != old.resolve():
            parser.error(f'Continuation source_run differs: {root}')
        if digest(root / 'preprocessor.joblib') != digest(old / 'preprocessor.joblib'):
            parser.error(f'Fitted preprocessing changed: {root}')
        with np.load(root / 'split.npz') as split, np.load(old / 'split.npz') as old_split:
            parts = [split[key] for key in ('train', 'validation', 'test')]
            if any(not np.array_equal(split[key], old_split[key]) for key in old_split.files):
                parser.error(f'Train/validation/test indices changed: {root}')
            joined = np.concatenate(parts)
            if len(joined) != len(np.unique(joined)):
                parser.error(f'Data splits overlap: {root}')
            train_rows = len(split['train'])
        counts = pd.read_csv(root / 'inputs' / 'class_counts.csv').set_index('class_index')['count']
        minority = int(counts.idxmin())
        old_report = read(old / 'final_report.json')
        new_report = read(root / 'final_report.json')
        if old_report['epochs'] != 30 or new_report['epochs'] != 100:
            parser.error(f'Expected fixed epoch budgets 30 and 100: {root}')
        curve = pd.read_csv(root / 'validation_scores.csv')
        history = pd.read_csv(root / 'losses.csv')
        old_curve = pd.read_csv(old / 'validation_scores.csv')
        old_history = pd.read_csv(old / 'losses.csv')
        if not history.iloc[:30].reset_index(drop=True).equals(old_history):
            parser.error(f'Epochs 1–30 loss history changed: {root}')
        if not curve[curve.epoch <= 30].reset_index(drop=True).equals(old_curve):
            parser.error(f'Epochs 1–30 validation history changed: {root}')
        if history.epoch.tolist() != list(range(1, 101)) or curve.epoch.tolist() != [1, *range(10, 101, 10)]:
            parser.error(f'Incomplete epoch/evaluation schedule: {root}')
        latest = torch.load(root / 'latest.pt', weights_only=True, map_location='cpu')
        original = torch.load(old / 'latest.pt', weights_only=True, map_location='cpu')
        batches = math.ceil(train_rows / manifest['batch_size'])
        if latest['epochs_trained'] != 100 or original['epochs_trained'] != 30:
            parser.error(f'Checkpoint epoch counters differ: {root}')
        for key in ('generator_optimizer', 'discriminator_optimizer'):
            if optimizer_steps(original, key) != {float(30 * batches)} or optimizer_steps(latest, key) != {float(100 * batches)}:
                parser.error(f'Unexpected Adam step counters: {root}/{key}')
        if old_report['real'] != new_report['real']:
            parser.error(f'Real-trained test reference changed: {root}')
        for budget, run, report, scores in ((30, old, old_report, old_curve), (100, root, new_report, curve)):
            first_max = int(scores.loc[scores.f1_macro.idxmax(), 'epoch'])
            if first_max != report['best_epoch']:
                parser.error(f'Checkpoint selection differs from first validation macro-F1 maximum: {run}')
            for checkpoint, key in (('best', 'validation_selected'), ('last', 'last')):
                epoch = first_max if checkpoint == 'best' else budget
                utility = report[key]
                validation = scores[scores.epoch == epoch].iloc[0]
                rows.append({'seed': seed, 'budget': budget, 'checkpoint': checkpoint, 'epoch': epoch,
                             'validation_macro_f1': float(validation.f1_macro),
                             **{metric: utility[metric] for metric in METRICS if metric != 'minority_recall'},
                             'minority_recall': recall(utility, minority),
                             **diversity(run / 'test' / checkpoint)})
        recalls = []
        for epoch in curve.epoch:
            folder = root / 'validation' / f'epoch_{int(epoch)}'
            recalls.append(recall(read(folder / 'utility.json'), minority))
            variations.append({'seed': seed, 'epoch': int(epoch), **diversity(folder)})
        curves.append(curve.assign(seed=seed, minority_recall=recalls))
        losses.append(history.assign(seed=seed))
        audit.append({'seed': seed, 'from_epoch': 30, 'through_epoch': 100,
                      'original_checkpoint_sha256': recorded_digest,
                      'final_checkpoint_sha256': digest(root / 'latest.pt'),
                      'loss_prefix_exact': True, 'validation_prefix_exact': True,
                      'split_exact': True, 'preprocessor_exact': True,
                      'adam_steps_before': 30 * batches, 'adam_steps_after': 100 * batches,
                      'optimizer_and_rng_resume_recorded': True, 'source_hashes_exact': True})

    final = pd.DataFrame(rows); validation = pd.concat(curves, ignore_index=True)
    history = pd.concat(losses, ignore_index=True); variation = pd.DataFrame(variations)
    aggregate = final.groupby(['budget', 'checkpoint'])[['validation_macro_f1', *METRICS, *DIVERSITY]].agg(['mean', 'std'])
    selected = final[final.checkpoint == 'best'].set_index(['seed', 'budget'])
    deltas = selected.xs(100, level='budget').select_dtypes(include='number') - selected.xs(30, level='budget').select_dtypes(include='number')
    out = args.output; out.mkdir(parents=True)
    for name, frame in {'final_scores': final, 'validation_scores': validation, 'losses': history,
                        'diversity': variation, 'continuation_audit': pd.DataFrame(audit),
                        'source_audit': pd.DataFrame(source_audit)}.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    aggregate.to_csv(out / 'aggregate.csv'); deltas.to_csv(out / 'selected_paired_deltas.csv')

    def trajectory(ax, frame, metric):
        for seed, part in frame.groupby('seed'):
            ax.plot(part.epoch, part[metric], alpha=.45, linewidth=1, label=f'Seed {seed}')
        grouped = frame.groupby('epoch')[metric]; mean = grouped.mean(); sd = grouped.std().fillna(0)
        ax.plot(mean.index, mean, color='black', linewidth=2, label='Mean')
        ax.fill_between(mean.index, mean - sd, mean + sd, color='black', alpha=.12)
        ax.axvline(30, color='tab:orange', linestyle='--', label='Continuation starts')
        ax.set(xlabel='Epoch', ylabel=metric); ax.grid(alpha=.2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, METRICS):
        trajectory(ax, validation, metric); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    axes[0, 0].axhline(.904, color='red', linestyle=':', label='90.4% majority reference')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Validation utility through 100 epochs: individual seeds and mean ± sample SD')
    fig.tight_layout(); fig.savefig(out / 'validation_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, DIVERSITY):
        trajectory(ax, variation, metric); ax.axhline(1, color='green', linestyle=':'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'diversity_curves.png', dpi=150); plt.close(fig)

    def cell(part, metric):
        return f'{part[metric].mean():.4f} ± {part[metric].std():.4f}'

    lines = ['# Conditional-moment continuation through 100 epochs', '',
             'Three exact continuations, seeds 42/43/44, from epoch 30 to total epoch 100. '
             'Weights, Adam states and RNG state were resumed; the split, preprocessing, circuit, '
             'output sampling, moment weight 10, learning rates and batch size were retained. '
             'Training uses the preserved source snapshot so later working-tree fixes do not alter the comparison.', '',
             '## Predictive utility', '',
             'Best checkpoints use the first maximum validation macro F1 observed through each budget. '
             'The 100-epoch search includes all original checkpoints. Fixed checkpoints are the actual '
             'epochs 30 and 100. Test scores do not choose checkpoints. Means ± sample SD across three seeds.', '',
             '| Budget | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (budget, checkpoint), part in final.groupby(['budget', 'checkpoint'], sort=True):
        lines.append('| ' + ' | '.join([str(budget), checkpoint, cell(part, 'validation_macro_f1'),
                                       *[cell(part, metric) for metric in METRICS]]) + ' |')
    lines += ['', '## Numeric variation and categories', '',
              '| Budget | Checkpoint | Numeric within-class std ratio | Category coverage |',
              '| ---: | --- | ---: | ---: |']
    for (budget, checkpoint), part in final.groupby(['budget', 'checkpoint']):
        lines.append('| ' + ' | '.join([str(budget), checkpoint, *[cell(part, metric) for metric in DIVERSITY]]) + ' |')
    lines += ['', 'The numeric ratio averages the smaller generated/real within-class standard-deviation '
              'ratio per numeric feature. Category coverage averages the fraction of observed training '
              'categories generated per feature. Neither metric certifies the full distribution.', '',
              '## Selected checkpoints by seed', '',
              '| Seed | Budget | Selected epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Minority recall |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in final[final.checkpoint == 'best'].itertuples():
        lines.append(f'| {row.seed} | {row.budget} | {row.epoch} | {row.validation_macro_f1:.4f} | '
                     f'{row.accuracy:.4f} | {row.f1_macro:.4f} | {row.roc_auc:.4f} | {row.minority_recall:.4f} |')
    old_best = final[(final.budget == 30) & (final.checkpoint == 'best')]
    new_best = final[(final.budget == 100) & (final.checkpoint == 'best')]
    lines += ['', '## What the longer budget changed', '',
              f'Mean validation-selected test accuracy changed {old_best.accuracy.mean():.1%} → '
              f'{new_best.accuracy.mean():.1%}; macro F1 changed {old_best.f1_macro.mean():.4f} → '
              f'{new_best.f1_macro.mean():.4f}; AUROC changed {old_best.roc_auc.mean():.4f} → '
              f'{new_best.roc_auc.mean():.4f}. These are reports of the frozen validation selections, '
              'not a rule to choose an epoch from its test score.', '',
              'More epochs need not improve utility. Validation-best F1 cannot decrease when earlier '
              'checkpoints remain candidates, but test utility and the final checkpoint can worsen. '
              'Accuracy should be read alongside minority recall and macro F1; the majority-only '
              'accuracy reference is 90.4%.', '',
              '[Validation curves](validation_curves.png) · [Diversity curves](diversity_curves.png) · '
              '[Per-seed scores](final_scores.csv) · [Paired changes](selected_paired_deltas.csv)', '',
              '## Verification and scope', '',
              'All 30-epoch loss and validation prefixes match exactly. Split arrays and serialized '
              'preprocessors match, retained source checkpoint hashes match continuation provenance, '
              'Adam step counters advance from 30 to 100 epochs, and preserved launch-source hashes '
              'match every run manifest. See [continuation_audit.csv](continuation_audit.csv) and '
              '[source_audit.csv](source_audit.csv).', '',
              'The 30-epoch experiment and these splits were already examined. This continuation is '
              'exploratory, not independent confirmation or a significance test. The mixed-type output '
              'and moment objective extend the supplied paper. Categorical generation includes classical '
              'Gumbel randomness; these curves do not establish a quantum advantage. No threshold '
              'calibration is applied in this report.', '',
              f'Original study: `{args.previous}`. Continuation study: `{args.study}`.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(aggregate.to_string())
    print('All continuation, source, split, history and checkpoint audits passed.')


if __name__ == '__main__':
    main()
