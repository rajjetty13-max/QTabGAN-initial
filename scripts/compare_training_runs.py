"""Compare completed extended_benchmark.py circuit trials with matching protocols."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qtabgan.evaluation import correlation_difference


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('runs', type=Path, nargs=2)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = [read(run/'manifest.json') for run in args.runs]
    for key in ['data_sha256', 'target', 'categorical', 'seed', 'batch_size', 'eval_every', 'protocol', 'source_sha256', 'versions', 'training_seed']:
        if manifests[0].get(key) != manifests[1].get(key): parser.error(f'Runs differ in {key}')
    import hashlib
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != manifests[0]['data_sha256']:
        parser.error('Input data does not match training data hash')
    splits = [np.load(run/'split.npz') for run in args.runs]
    for key in ['train','validation','test']:
        if not np.array_equal(splits[0][key], splits[1][key]): parser.error(f'{key} rows differ')
    reports = [read(run/'final_report.json') for run in args.runs]
    if reports[0]['epochs'] != reports[1]['epochs']: parser.error('Epoch budgets differ')
    if reports[0]['real'] != reports[1]['real']: parser.error('Real-trained references differ')
    if args.output.exists(): parser.error('Choose a fresh comparison output folder')
    args.output.mkdir(parents=True)
    raw = pd.read_csv(args.data).iloc[splits[0]['train']]
    rows = []
    for run, manifest, report in zip(args.runs, manifests, reports):
        pre = joblib.load(run/'preprocessor.joblib')
        variant = manifest['ansatz'] + (' + repeated H' if manifest.get('hadamards_per_layer', False) else '') + (' + D(x,y)' if manifest.get('condition_discriminator', False) else ' + D(x)')
        for checkpoint, key in [('last','last'), ('best','validation_selected')]:
            folder = run/'test'/checkpoint
            numeric = pd.read_csv(folder/'raw_numeric_diagnostics.csv')
            coverage = pd.read_csv(folder/'category_coverage.csv')
            fake = pd.read_csv(folder/'synthetic.csv')
            rows.append({'ansatz': variant, 'checkpoint': checkpoint,
                         'epoch': report['epochs'] if checkpoint == 'last' else report['best_epoch'],
                         **{k:v for k,v in report[key].items() if k != 'confusion_matrix'},
                         'mean_raw_numeric_ks': float(np.mean([ks_2samp(raw[c], fake[c]).statistic for c in pre.numerical_columns])),
                         'raw_numeric_correlation_error': correlation_difference(raw[pre.numerical_columns].to_numpy(),fake[pre.numerical_columns].to_numpy()),
                         'mean_numeric_std_ratio': float(numeric.std_ratio.mean()),
                         'mean_minimum_within_class_std_ratio': float(numeric.min_within_class_std_ratio.mean()),
                         'mean_category_coverage': float(coverage.training_category_coverage.mean())})
    table = pd.DataFrame(rows); table.to_csv(args.output/'summary.csv',index=False)
    fig, axes = plt.subplots(1,4,figsize=(16,4))
    for run, manifest in zip(args.runs, manifests):
        scores = pd.read_csv(run/'validation_scores.csv')
        for ax, metric in zip(axes,['accuracy','f1_macro','roc_auc','average_precision']):
            ax.plot(scores.epoch,scores[metric],marker='o',label=manifest['ansatz'] + (' + repeated H' if manifest.get('hadamards_per_layer', False) else '') + (' + D(x,y)' if manifest.get('condition_discriminator', False) else ' + D(x)'))
            ax.set(title=metric,xlabel='Epoch',ylim=(0,1)); ax.legend()
    fig.tight_layout();fig.savefig(args.output/'validation_comparison.png');plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for run, manifest in zip(args.runs, manifests):
        history = pd.read_csv(run/'losses.csv')
        for ax, metric in zip(axes,['generator_loss','discriminator_loss','quantum_gradient_norm']):
            ax.plot(history.epoch,history[metric],label=manifest['ansatz'] + (' + repeated H' if manifest.get('hadamards_per_layer', False) else '') + (' + D(x,y)' if manifest.get('condition_discriminator', False) else ' + D(x)'))
            ax.set(title=metric,xlabel='Epoch');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'loss_comparison.png');plt.close(fig)
    truth = pd.read_csv(args.runs[0]/'test/real/predictions.csv').truth
    majority = float(truth.value_counts(normalize=True).max())
    lines = ['# Matched circuit training test', '',
             f'Both variants trained for {reports[0]["epochs"]} epochs on the same full Loan split and seed {manifests[0]["seed"]}.',
             'Checkpoint selection uses highest validation macro F1; the following scores are on the shared final real test set.', '',
             '| Circuit | Checkpoint | Epoch | Accuracy | Macro F1 | AUROC | Average precision | Mean raw numeric KS |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["ansatz"]} | {row["checkpoint"]} | {row["epoch"]} | {row["accuracy"]:.4f} | {row["f1_macro"]:.4f} | {row["roc_auc"]:.4f} | {row["average_precision"]:.4f} | {row["mean_raw_numeric_ks"]:.4f} |')
    real = reports[0]['real']
    lines += ['', f'Real-trained reference: accuracy {real["accuracy"]:.4f}, macro F1 {real["f1_macro"]:.4f}, AUROC {real["roc_auc"]:.4f}, average precision {real["average_precision"]:.4f}.',
              f'Majority-class test accuracy: {majority:.4f}. One seed and a short budget are exploratory evidence, not proof of a reliable winner or a paper reproduction.', '',
              '## Diversity checks', '', '| Circuit | Checkpoint | Mean numeric std / real std | Mean minimum within-class std / real std | Mean category coverage |',
              '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["ansatz"]} | {row["checkpoint"]} | {row["mean_numeric_std_ratio"]:.5f} | {row["mean_minimum_within_class_std_ratio"]:.5f} | {row["mean_category_coverage"]:.4f} |')
    lines += ['', 'Ratios near zero indicate collapse; a ratio near one alone does not establish a distribution match. Lower numeric KS is better. Category coverage is an unweighted average across categorical features.', '',
              '[Validation trajectories](validation_comparison.png) · [Losses and quantum gradients](loss_comparison.png) · [Complete numbers](summary.csv)', '',
              'Each source run contains test/real, test/last and test/best confusion matrices, ROC and precision–recall plots, predictions, synthetic rows and raw diagnostics. Runs:', '']
    lines.extend(f'- `{run}`' for run in args.runs)
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(table.to_string(index=False))


if __name__ == '__main__': main()
