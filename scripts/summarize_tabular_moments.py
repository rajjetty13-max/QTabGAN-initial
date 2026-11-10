"""Audit and summarize paired mixed-type-output and conditional-moment trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


VARIANTS = ('control', 'typed_output', 'moment_matched')
METRICS = ('accuracy', 'f1_macro', 'roc_auc', 'average_precision', 'minority_recall')
DIVERSITY = ('within_class_std_ratio', 'category_coverage')


def read(path):
    return json.loads(path.read_text())


def recall(utility, minority):
    matrix = np.asarray(utility['confusion_matrix'])
    total = matrix[minority].sum()
    return float(matrix[minority, minority] / total) if total else 0.0


def diversity(folder):
    numeric = pd.read_csv(folder / 'raw_numeric_diagnostics.csv')
    categorical = pd.read_csv(folder / 'category_coverage.csv')
    return {'within_class_std_ratio': float(numeric.min_within_class_std_ratio.mean()),
            'category_coverage': float(categorical.training_category_coverage.mean())}


def schema(preprocessor):
    categories = ([] if preprocessor.encoder is None else
                  [category.tolist() for category in preprocessor.encoder.categories_])
    return {'numerical_columns': preprocessor.numerical_columns,
            'categorical_columns': preprocessor.categorical_columns,
            'output_columns': preprocessor.output_columns,
            'categories': categories, 'noise_std': preprocessor.noise_std,
            **{name: getattr(preprocessor, name) for name in
               ('q1', 'q99', 'mean', 'std', 'min_value', 'max_value')}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    # Changes must be named and explained; they are never silently ignored.
    parser.add_argument('--allow-source-change', action='append', default=[], metavar='FILE=REASON')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh analysis folder')
    allowed_sources = {}
    for item in args.allow_source_change:
        name, separator, reason = item.partition('=')
        if not separator or not name or not reason.strip():
            parser.error('--allow-source-change requires FILE=REASON')
        allowed_sources[name] = reason

    groups = {}; omitted = []
    control_roots = sorted((args.study / 'runs' / 'control').glob('seed_*'))
    if not control_roots or any(not (root / 'final_report.json').exists() for root in control_roots):
        parser.error('A fully completed control group is required')
    seeds = {int(root.name.removeprefix('seed_')) for root in control_roots}
    for variant in VARIANTS:
        roots = sorted((args.study / 'runs' / variant).glob('seed_*'))
        complete = [root for root in roots if (root / 'final_report.json').exists()]
        completed_seeds = {int(root.name.removeprefix('seed_')) for root in complete}
        if not roots or len(complete) != len(roots) or completed_seeds != seeds:
            omitted.append(f'{variant}: completed seeds {sorted(completed_seeds)}; '
                           f'expected the full control seed set {sorted(seeds)}')
            continue
        groups[variant] = complete

    roots = [root for group in groups.values() for root in group]
    manifests = {root: read(root / 'manifest.json') for root in roots}
    reference = manifests[control_roots[0]]
    fixed = ('data_sha256', 'ansatz', 'hadamards_per_layer', 'condition_discriminator',
             'n_layers', 'target', 'categorical', 'batch_size', 'eval_every', 'protocol',
             'versions', 'quantum_init_gain', 'lr_generator', 'lr_discriminator',
             'categorical_temperature', 'paper_reproduction')
    source_reference = {}; source_observations = {}; paired = {}; budgets = set()
    rows = []; real_rows = []; curves = []; losses = []; variation = []; audit_rows = []
    variant_settings = {}; majority_fractions = []
    helper = 'qtabgan/regularization.py'
    for root in roots:
        manifest = manifests[root]; variant = root.parent.name; seed = manifest['seed']
        if root.name != f'seed_{seed}':
            parser.error(f'Seed directory and manifest differ: {root}')
        for key in fixed:
            if manifest.get(key) != reference.get(key):
                parser.error(f'Unexpected change in {key}: {root}')
        sources = manifest['source_sha256']
        if set(sources) - {helper} != set(reference['source_sha256']) - {helper}:
            parser.error(f'Source-file sets differ outside the optional moment helper: {root}')
        for name, digest in sources.items():
            source_observations.setdefault(name, set()).add(digest)
            if name in source_reference and digest != source_reference[name] and name not in allowed_sources:
                parser.error(f'Source differs for {name}: {root}; inspect the change, then '
                             'record a legitimate reason with --allow-source-change FILE=REASON')
            source_reference.setdefault(name, digest)
        if manifest['moment_loss_weight'] > 0 and helper not in sources:
            parser.error(f'Moment run does not record its regularization source: {root}')
        settings = (manifest['tabular_output'], manifest['moment_loss_weight'])
        if variant in variant_settings and variant_settings[variant] != settings:
            parser.error(f'Output/loss settings differ within {variant}')
        variant_settings[variant] = settings
        if ((variant == 'control' and settings != (False, 0.0)) or
                (variant == 'typed_output' and settings != (True, 0.0)) or
                (variant == 'moment_matched' and (not settings[0] or settings[1] <= 0))):
            parser.error(f'Variant name does not match its output/loss settings: {root}')

        split_file = np.load(root / 'split.npz')
        split = {name: split_file[name] for name in ('train', 'validation', 'test')}
        split_file.close()
        indices = np.concatenate(list(split.values()))
        if len(indices) != len(np.unique(indices)):
            parser.error(f'Train/validation/test rows overlap: {root}')
        preprocessor = joblib.load(root / 'preprocessor.joblib')
        fitted_schema = schema(preprocessor)
        flow = read(root / 'inputs' / 'data_flow.json')
        initialization = (manifest['classical_init_seed'], manifest['training_seed'])
        if seed in paired:
            earlier = paired[seed]
            if any(not np.array_equal(split[name], earlier['split'][name]) for name in split):
                parser.error(f'Paired split rows differ at seed {seed}')
            if initialization != earlier['initialization']:
                parser.error(f'Initialization/training seeds differ at seed {seed}')
            if fitted_schema != earlier['schema'] or flow != earlier['flow']:
                parser.error(f'Fitted schema, preprocessing or class order differ at seed {seed}')
        else:
            paired[seed] = dict(split=split, initialization=initialization,
                                schema=fitted_schema, flow=flow)
        class_counts = pd.read_csv(root / 'inputs' / 'class_counts.csv').set_index('class_index')['count']
        minority = int(class_counts.idxmin())
        majority_fractions.append(float(class_counts.max() / class_counts.sum()))
        report = read(root / 'final_report.json'); epochs = int(report['epochs']); budgets.add(epochs)
        scores = pd.read_csv(root / 'validation_scores.csv')
        loss = pd.read_csv(root / 'losses.csv')
        expected_epochs = sorted({1, epochs, *range(manifest['eval_every'], epochs + 1, manifest['eval_every'])})
        if scores.epoch.tolist() != expected_epochs or loss.epoch.tolist() != list(range(1, epochs + 1)):
            parser.error(f'Epoch or evaluation schedule is incomplete: {root}')
        selected_epoch = int(scores.loc[scores.f1_macro.idxmax(), 'epoch'])
        if report['best_epoch'] != selected_epoch:
            parser.error(f'Best checkpoint does not use first maximum validation macro F1: {root}')
        validation_recalls = []
        for epoch in scores.epoch:
            folder = root / 'validation' / f'epoch_{int(epoch)}'
            utility = read(folder / 'utility.json')
            validation_recalls.append(recall(utility, minority))
            variation.append(dict(seed=seed, variant=variant, epoch=int(epoch), **diversity(folder)))
        curves.append(scores.assign(seed=seed, variant=variant, minority_recall=validation_recalls))
        losses.append(loss.assign(seed=seed, variant=variant))
        real_rows.append(dict(seed=seed, **{key: report['real'].get(key, float('nan'))
                                          for key in METRICS if key != 'minority_recall'},
                              minority_recall=recall(report['real'], minority)))
        for checkpoint, report_key in (('best', 'validation_selected'), ('last', 'last')):
            epoch = selected_epoch if checkpoint == 'best' else epochs
            validation = scores.loc[scores.epoch == epoch].iloc[0]
            utility = report[report_key]
            rows.append(dict(seed=seed, variant=variant, checkpoint=checkpoint, epoch=epoch,
                             validation_macro_f1=float(validation.f1_macro),
                             **{key: utility.get(key, float('nan')) for key in METRICS
                                if key != 'minority_recall'}, minority_recall=recall(utility, minority),
                             **diversity(root / 'test' / checkpoint)))
        audit_rows.append(dict(variant=variant, seed=seed, epochs=epochs,
                               tabular_output=settings[0], moment_loss_weight=settings[1],
                               train_rows=len(split['train']), validation_rows=len(split['validation']),
                               test_rows=len(split['test']), numerical_dim=len(preprocessor.numerical_columns),
                               categorical_dims=json.dumps([len(c) for c in fitted_schema['categories']]),
                               classical_init_seed=initialization[0], training_seed=initialization[1],
                               source_directory=str(root)))
    if len(budgets) != 1:
        parser.error('Epoch budgets differ')
    snapshot = args.study / 'source'
    source_audit = []
    for name, digests in sorted(source_observations.items()):
        snapshot_file = snapshot / name
        snapshot_digest = None
        if snapshot.exists():
            if not snapshot_file.is_file():
                parser.error(f'Launch-source snapshot is missing {name}')
            snapshot_digest = hashlib.sha256(snapshot_file.read_bytes()).hexdigest()
            if snapshot_digest not in digests:
                parser.error(f'Launch-source snapshot does not match a recorded hash: {name}')
        source_audit.append(dict(source_file=name, recorded_sha256=json.dumps(sorted(digests)),
                                 launch_snapshot_sha256=snapshot_digest))
    final = pd.DataFrame(rows); curve_frame = pd.concat(curves, ignore_index=True)
    loss_frame = pd.concat(losses, ignore_index=True); diversity_frame = pd.DataFrame(variation)
    real = pd.DataFrame(real_rows).drop_duplicates()
    if real.seed.duplicated().any():
        parser.error('Real-trained reference scores differ within a paired seed')
    best = final[final.checkpoint == 'best']
    validation_ranking = best.groupby('variant').validation_macro_f1.mean().sort_values(ascending=False)
    winner = validation_ranking.index[0]
    out = args.output; out.mkdir(parents=True)
    tables = {'final_scores': final, 'validation_scores': curve_frame, 'losses': loss_frame,
              'diversity': diversity_frame, 'real_test_scores': real,
              'protocol_audit': pd.DataFrame(audit_rows), 'source_audit': pd.DataFrame(source_audit)}
    for name, frame in tables.items():
        frame.to_csv(out / f'{name}.csv', index=False)
    aggregate = final.groupby(['variant', 'checkpoint'])[['validation_macro_f1', *METRICS, *DIVERSITY]].agg(['mean', 'std'])
    aggregate.to_csv(out / 'aggregate.csv')
    validation_ranking.rename('mean_validation_macro_f1').to_csv(out / 'validation_ranking.csv')

    def trajectory(ax, frame, metric):
        for name, part in frame.groupby('variant', sort=False):
            grouped = part.groupby('epoch')[metric]; mean = grouped.mean(); sd = grouped.std().fillna(0)
            ax.plot(mean.index, mean, marker='.', label=name)
            ax.fill_between(mean.index, mean - sd, mean + sd, alpha=.15)
        ax.set(xlabel='Epoch', ylabel=metric); ax.grid(alpha=.2); ax.legend(fontsize=8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, METRICS):
        trajectory(ax, curve_frame, metric); ax.set_ylim(0, 1)
    axes[0, 0].axhline(np.mean(majority_fractions), color='red', linestyle='--', label='Majority reference')
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].axhline(.5, color='gray', linestyle='--')
    axes.flat[-1].set_visible(False)
    fig.suptitle('Validation utility: mean ± sample SD over paired seeds')
    fig.tight_layout(); fig.savefig(out / 'validation_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, DIVERSITY):
        trajectory(ax, diversity_frame, metric)
    fig.tight_layout(); fig.savefig(out / 'diversity_curves.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ('generator_loss', 'discriminator_loss', 'quantum_gradient_norm')):
        trajectory(ax, loss_frame, metric)
    fig.tight_layout(); fig.savefig(out / 'loss_curves.png', dpi=150); plt.close(fig)

    def cell(part, metric):
        mean = part[metric].mean(); sd = part[metric].std()
        return f'{mean:.4f} ± {sd:.4f}' if np.isfinite(sd) else f'{mean:.4f} (one seed)'

    control_f1 = validation_ranking['control']
    lines = ['# Mixed-type output and conditional-moment experiment', '',
             f'Completed groups: {", ".join(groups)}. Seeds: {sorted(seeds)}. '
             f'Each run completed {next(iter(budgets))} epochs. Paired split arrays, fitted preprocessing/schema, '
             'class order, initialization/training seeds, learning rates, circuit, gain and evaluation schedules were checked.', '',
             '| Variant | Typed output | Moment weight |', '| --- | --- | ---: |',
             *[f'| {name} | {settings[0]} | {settings[1]:g} |' for name, settings in variant_settings.items()], '',
             f'All groups use G/D learning rates {reference["lr_generator"]:g}/{reference["lr_discriminator"]:g}, '
             f'quantum initialization gain {reference["quantum_init_gain"]:g}, '
             f'{reference["n_layers"]} layers of `{reference["ansatz"]}`, and categorical temperature '
             f'{reference["categorical_temperature"]:g}.', '',
             '## Decision from validation', '',
             f'The highest mean validation-selected macro F1 is **{winner}** '
             f'({validation_ranking[winner]:.4f}); control is {control_f1:.4f}. '
             'This ranking uses validation scores only. The test results below did not determine the ranking.', '',
             'Checkpoints use the first maximum validation macro F1 within each run. '
             'Best and final checkpoints are both retained. Means ± sample SD describe these seeds, '
             'without establishing statistical significance.', '',
             '## Held-out predictive utility', '',
             '| Variant | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (variant, checkpoint), part in final.groupby(['variant', 'checkpoint'], sort=False):
        lines.append('| ' + ' | '.join([variant, checkpoint, cell(part, 'validation_macro_f1'),
                                       *[cell(part, metric) for metric in METRICS]]) + ' |')
    lines += ['', f'The training majority-class proportion is {100 * np.mean(majority_fractions):.1f}%. '
              'Predicting only that class reaches this accuracy without minority discrimination. '
              'Minority recall uses the least frequent training class and its matching confusion-matrix row.', '',
              'Real-trained classifier test reference: ' + ', '.join(f'{metric} {cell(real, metric)}' for metric in METRICS) + '.', '',
              '## Generated-data variation', '',
              '| Variant | Checkpoint | Numeric within-class std ratio | Category coverage |',
              '| --- | --- | ---: | ---: |']
    for (variant, checkpoint), part in final.groupby(['variant', 'checkpoint'], sort=False):
        lines.append('| ' + ' | '.join([variant, checkpoint, *[cell(part, metric) for metric in DIVERSITY]]) + ' |')
    lines += ['', 'The numeric ratio averages the smaller class-specific generated/real standard-deviation ratio '
              'across raw numeric features; it does not certify every class-feature pair. '
              'Coverage averages the fraction of observed training categories generated per categorical feature.', '',
              '## Individual runs', '',
              '| Variant | Seed | Checkpoint | Epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP | Minority recall |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in final.itertuples(index=False):
        values = [row.variant, str(row.seed), row.checkpoint, str(row.epoch),
                  f'{row.validation_macro_f1:.4f}', *[f'{getattr(row, metric):.4f}' for metric in METRICS]]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '[Validation trajectories](validation_curves.png) · [Variation trajectories](diversity_curves.png) · '
              '[Loss trajectories](loss_curves.png) · [All per-seed metrics](final_scores.csv) · '
              '[Protocol audit](protocol_audit.csv)', '', '## Interpretation and scope', '',
              'These splits were examined in earlier work; this is exploratory evidence, not independent confirmation. '
              'Additional output sampling changes the random-number stream even though paired runs start '
              'from matching training seeds. Identical subsequent latent/minibatch draws are not claimed.', '',
              'Typed output adds a bounded numeric sigmoid and straight-through Gumbel categorical samples. '
              'It introduces classical randomness at training and inference. Conditional moment matching adds '
              'an objective based only on training rows. Both are explicit extensions to the supplied paper, '
              'which specifies a linear mapper output. The retained final-H readout is also an extension; '
              'analytic circuit training differs from its finite-shot estimator. These results do not establish '
              'a faithful reproduction or a quantum advantage.', '',
              'Generator losses include the moment penalty when enabled and therefore are not directly '
              'comparable across objectives. Additional Gumbel randomness can raise apparent diversity '
              'without proving that the quantum input supplies that diversity.', '',
              '## Source audit', '',
              'Shared recorded source hashes were checked. The regularization helper may be absent from '
              'control manifests because those runs started before that file arrived; all recorded helper hashes must agree.']
    changed = {name: values for name, values in source_observations.items() if len(values) > 1}
    if changed:
        lines += ['', 'Explicitly reviewed source differences:']
        lines += [f'- `{name}`: {allowed_sources[name]}' for name in sorted(changed)]
    else:
        lines += ['', 'All shared source hashes match.']
    if snapshot.exists():
        lines += ['', f'The exact launch source is preserved in `{snapshot}`. Its hashes were '
                  'verified against the manifests and are listed in [source_audit.csv](source_audit.csv). '
                  'Use this snapshot to reproduce the experiment; the live working tree may contain later fixes.']
    if omitted:
        lines += ['', '## Omitted groups', '', *[f'- {reason}.' for reason in omitted]]
    lines += ['', 'Source runs:', '', *[f'- `{root}`' for root in roots]]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Validation-ranked choice: {winner} ({validation_ranking[winner]:.4f})')
    print(aggregate.to_string())


if __name__ == '__main__':
    main()
