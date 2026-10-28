"""Validation-selected, restartable QTabGAN experiment. See EXPERIMENT_WORKFLOW.md."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import pennylane as qml
import importlib.metadata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from benchmark import evaluate, distribution, save_json
from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.preprocessing import TabularPreprocessor
from qtabgan.utils import set_seed


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]]}


def restore_rng(s):
    random.setstate(s['python']); torch.set_rng_state(s['torch'])
    n = s['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))


@contextmanager
def evaluation_rng(seed):
    state = rng_state()
    try:
        set_seed(seed)
        yield
    finally:
        restore_rng(state)


def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp); tmp.replace(path)


def heatmap(frame, path, title):
    frame.to_csv(path.with_suffix('.csv'))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(frame, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(frame)), frame.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(frame)), frame.index, fontsize=7)
    ax.set_title(title); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def input_audit(frame, encoded, pre, y, labels, out, condition_discriminator=False):
    out.mkdir(exist_ok=True)
    frame.describe(include='all').to_csv(out / 'input_summary.csv')
    pd.DataFrame(encoded, columns=pre.output_columns).describe().to_csv(out / 'encoded_summary.csv')
    pd.Series(y).value_counts().rename_axis('class_index').to_csv(out / 'class_counts.csv')
    pd.DataFrame([{'feature': c, 'categories': frame[c].nunique()} for c in pre.categorical_columns]).to_csv(out / 'category_counts.csv', index=False)
    nums = frame[pre.numerical_columns]
    heatmap(nums.corr(), out / 'input_correlations.png', 'Training numerical feature correlations')
    fig, axes = plt.subplots(max(1, (len(nums.columns)+2)//3), 3, figsize=(12, 3*max(1,(len(nums.columns)+2)//3)), squeeze=False)
    for ax, c in zip(axes.flat, nums.columns):
        ax.hist(nums[c], bins=35); ax.set_title(c)
    for ax in list(axes.flat)[len(nums.columns):]: ax.set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'input_histograms.png'); plt.close(fig)
    idx = np.argsort(encoded.var(axis=0))[-20:]
    heatmap(pd.DataFrame(encoded[:, idx], columns=np.array(pre.output_columns)[idx]).corr(), out / 'encoded_correlations.png', '20 highest-variance encoded features')
    save_json(out / 'data_flow.json', {'train_rows': len(frame), 'original_features': len(frame.columns),
        'encoded_features': encoded.shape[1], 'label_order': [str(x) for x in labels],
        'numeric_encoding': 'Train 1st/99th percentile clipping; standardize; scale to [0, pi]',
        'categorical_encoding': 'Train-fitted one-hot; unseen categories become all zero',
        'quantum_input': 'Independent uniform random angles in [0, pi], NOT real data rows',
        'generator': '8 random angles -> 256 quantum probabilities + class one-hot -> classical mapper -> encoded features',
        'discriminator': ('Encoded features plus matching one-hot class labels: D(x,y)' if condition_discriminator else 'Encoded real or generated features only; does not see class labels')})


def raw_diagnostics(real, fake, numeric, categorical, y, out):
    rows = []
    for c in numeric:
        a, b = real[c].to_numpy(), fake[c].to_numpy()
        within = []
        for label in np.unique(y):
            mask = y == label
            if a[mask].std() > 0: within.append(float(b[mask].std()/a[mask].std()))
        rows.append({'feature': c, 'real_mean': a.mean(), 'synthetic_mean': b.mean(),
                     'std_ratio': float(b.std()/a.std()) if a.std() else None,
                     'min_within_class_std_ratio': min(within) if within else None,
                     'outside_training_range_fraction': float(((b<a.min())|(b>a.max())).mean()),
                     'negative_fraction': float((b<0).mean())})
    pd.DataFrame(rows).to_csv(out / 'raw_numeric_diagnostics.csv', index=False)
    pd.DataFrame([{'feature': c, 'training_categories': real[c].nunique(), 'generated_categories': fake[c].nunique(),
                   'training_category_coverage': len(set(real[c]) & set(fake[c]))/real[c].nunique()}
                  for c in categorical]).to_csv(out / 'category_coverage.csv', index=False)
    heatmap(fake[numeric].corr(), out / 'generated_correlations.png', 'Generated numerical feature correlations')
    heatmap((real[numeric].corr()-fake[numeric].corr()).abs(), out / 'correlation_error.png', 'Absolute correlation error (undefined stays blank)')


def plot_history(history, out):
    pd.DataFrame(history).to_csv(out / 'losses.csv', index=False)
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in ['generator_loss', 'discriminator_loss']: axes[0].plot(h.epoch, h[c], label=c)
    axes[0].legend(); axes[0].set(xlabel='Epoch', ylabel='BCE loss')
    axes[1].plot(h.epoch, h.quantum_gradient_norm); axes[1].set(xlabel='Epoch', ylabel='Quantum gradient norm')
    fig.tight_layout(); fig.savefig(out / 'losses.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--target', required=True)
    p.add_argument('--categorical', nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True); p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--condition-discriminator', action='store_true')
    p.add_argument('--hadamards-per-layer', action='store_true')
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--classical-init-seed', type=int)
    p.add_argument('--ansatz', choices=['paper', 'reuploading', 'paper_readout'], default='paper')
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=100); p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs,args.batch_size,args.eval_every,args.layers) < 1: p.error('Epochs, batch size and evaluation interval must be positive')
    torch.set_num_threads(1)
    out = args.output
    if out.exists() and not args.resume: p.error('Output exists. Choose a new directory or use --resume.')
    if args.resume and not (out/'latest.pt').exists(): p.error('--resume requires latest.pt')
    df = pd.read_csv(args.data)
    if df.isna().any().any(): p.error('Missing values must be handled explicitly')
    numeric = [c for c in df.columns if c != args.target and c not in args.categorical]
    if not numeric: p.error('At least one numeric feature is required for these diagnostics')
    y, labels = pd.factorize(df[args.target], sort=True)
    features = df.drop(columns=args.target)
    manifest = {'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                'hadamards_per_layer': args.hadamards_per_layer, 'condition_discriminator': args.condition_discriminator, 'training_seed': args.seed+20000,
                'n_layers': args.layers, 'classical_init_seed': args.classical_init_seed,
                'ansatz': args.ansatz, 'target': args.target, 'categorical': args.categorical, 'seed': args.seed,
                'batch_size': args.batch_size, 'eval_every': args.eval_every,
                'protocol': 'stratified 60% train / 20% validation / 20% test; validation macro F1 selection',
                'paper_reproduction': False,
                'versions': {name: importlib.metadata.version(name) for name in ['torch', 'pennylane', 'numpy', 'pandas', 'scikit-learn']},
                'source_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                                  [Path('extended_benchmark.py'), Path('benchmark.py'), *sorted(Path('qtabgan').glob('*.py'))]}}
    if args.resume:
        if json.loads((out/'manifest.json').read_text()) != manifest: p.error('Source, data or protocol differs from saved manifest; start a new run')
        state = torch.load(out/'latest.pt', weights_only=True)
        split = np.load(out/'split.npz'); tr, va, te = [split[k] for k in ['train','validation','test']]
        pre = joblib.load(out/'preprocessor.joblib')
    else:
        out.mkdir(parents=True)
        save_json(out/'manifest.json', manifest)
        trva, te = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=args.seed)
        tr, va = train_test_split(trva, test_size=.25, stratify=y[trva], random_state=args.seed)
        np.savez(out/'split.npz', train=tr, validation=va, test=te)
        pre = TabularPreprocessor(numeric, args.categorical, noise_std=0)
        pre.fit(features.iloc[tr]); joblib.dump(pre, out/'preprocessor.joblib')
        state = None
    x, validation, test = [pre.transform(features.iloc[idx]) for idx in [tr, va, te]]
    condition = np.eye(len(labels), dtype=np.float32)[y[tr]]
    config = QTabGANConfig(n_layers=args.layers, classical_init_seed=args.classical_init_seed, ansatz=args.ansatz, hadamards_per_layer=args.hadamards_per_layer, condition_discriminator=args.condition_discriminator, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    model = QTabGAN(x.shape[1], len(labels), config)
    if not args.resume:
        core = model.quantum_generator
        z = torch.zeros(config.n_qubits)
        (out/'training_circuit.txt').write_text(qml.draw(core.circuit)(z, core.weights))
        fig, _ = qml.draw_mpl(core.circuit)(z, core.weights)
        fig.savefig(out/'training_circuit.png', bbox_inches='tight'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(14, 3)); ax.axis('off')
        stages = [f'{config.n_qubits} random angles\nUniform [0, pi]',
                  f'{config.n_qubits}-qubit circuit\n{2**config.n_qubits} probabilities',
                  f'+ {len(labels)} class indicators\nClassical mapper',
                  f'{x.shape[1]} generated\nencoded features', ('Discriminator D(x,y)\nfeatures + class' if args.condition_discriminator else 'Discriminator\nreal vs generated')]
        for j, text in enumerate(stages):
            ax.text(.1+.2*j, .5, text, ha='center', va='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue'), fontsize=9)
            if j < len(stages)-1:
                ax.annotate('', xy=(.2+.2*j,.5), xytext=(.18+.2*j,.5), xycoords='axes fraction', arrowprops=dict(arrowstyle='->'))
        fig.savefig(out/'training_flow.png', bbox_inches='tight'); plt.close(fig)
    history, scores, best = [], [], -1.
    start_epoch = 0
    if state:
        model.load_state_dict(state['state_dict'])
        gp = list(model.quantum_generator.parameters()) + list(model.mapper.parameters())
        model._generator_optimizer = torch.optim.Adam(gp, lr=config.lr_generator)
        model._discriminator_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=config.lr_discriminator)
        model._generator_optimizer.load_state_dict(state['generator_optimizer'])
        model._discriminator_optimizer.load_state_dict(state['discriminator_optimizer'])
        history, scores, best, start_epoch = state['history'], state['validation_scores'], state['best_score'], state['epochs_trained']
        restore_rng(state['rng'])
    else:
        with evaluation_rng(args.seed): input_audit(features.iloc[tr], x, pre, y[tr], labels, out/'inputs', args.condition_discriminator)
        # Separate initialization draws from minibatch/latent randomness for architecture ablations.
        set_seed(args.seed+20000)
    if args.epochs < start_epoch: p.error('Requested epochs are below the saved epoch')
    def snapshot(epoch):
        return {'state_dict': model.state_dict(), 'config': asdict(config), 'epochs_trained': epoch,
                'generator_optimizer': model._generator_optimizer.state_dict(),
                'discriminator_optimizer': model._discriminator_optimizer.state_dict(),
                'rng': rng_state(), 'history': history, 'validation_scores': scores, 'best_score': best,
                'data_dim': x.shape[1], 'condition_dim': len(labels)}
    def assess(folder, heldout, truth):
        folder.mkdir(exist_ok=True, parents=True)
        with evaluation_rng(args.seed + 10000):
            raw = pre.inverse_transform(model.generate(condition))
            fake = pre.transform(raw)
            raw.assign(**{args.target: np.array(labels)[y[tr]]}).to_csv(folder/'synthetic.csv', index=False)
            utility = evaluate(fake, y[tr], heldout, truth, folder, labels, args.seed)
            fidelity = distribution(x, fake, pre.output_columns, folder)
            raw_diagnostics(features.iloc[tr], raw, numeric, args.categorical, y[tr], folder)
            save_json(folder/'metrics.json', {'utility': utility, 'fidelity': fidelity})
        return utility
    started = time.monotonic()
    def callback(local_epoch, row):
        nonlocal best
        epoch = start_epoch + local_epoch
        history.append({'epoch': epoch, **row})
        # Save losses every epoch so an interrupted process retains diagnostics.
        pd.DataFrame(history).to_csv(out/'losses.csv', index=False)
        print(f'Epoch {epoch}/{args.epochs}: G={row["generator_loss"]:.5f} D={row["discriminator_loss"]:.5f} quantum_grad={row["quantum_gradient_norm"]:.6g}', flush=True)
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            utility = assess(out/'validation'/f'epoch_{epoch}', validation, y[va])
            scores.append({'epoch': epoch, **{k: v for k,v in utility.items() if k != 'confusion_matrix'}})
            if utility['f1_macro'] > best:
                best = utility['f1_macro']; atomic_save(snapshot(epoch), out/'best.pt')
            pd.DataFrame(scores).to_csv(out/'validation_scores.csv', index=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            for metric in ['accuracy', 'f1_macro', 'roc_auc', 'average_precision']:
                if metric in scores[-1]: ax.plot([r['epoch'] for r in scores], [r[metric] for r in scores], marker='o', label=metric)
            ax.set(xlabel='Epoch', ylabel='Validation score', ylim=(0,1)); ax.legend()
            fig.tight_layout(); fig.savefig(out/'validation_trajectory.png'); plt.close(fig)
            print(f'Validation epoch {epoch}: {utility}', flush=True)
        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            plot_history(history, out); atomic_save(snapshot(epoch), out/'latest.pt')
    if args.epochs > start_epoch:
        model.config.epochs = args.epochs-start_epoch
        model.fit(x, condition, verbose=False, epoch_callback=callback)
        model.config.epochs = args.epochs
        atomic_save(snapshot(args.epochs), out/'latest.pt')
    # Final holdout is used only after training. Resume extensions must be treated as a new exploratory study.
    teacher = out/'test'/'real'; teacher.mkdir(parents=True, exist_ok=True)
    baseline = evaluate(x, y[tr], test, y[te], teacher, labels, args.seed)
    last = assess(out/'test'/'last', test, y[te])
    best_state = torch.load(out/'best.pt', weights_only=True)
    model.load_state_dict(best_state['state_dict'])
    selected = assess(out/'test'/'best', test, y[te])
    save_json(out/'final_report.json', {'epochs': args.epochs, 'best_epoch': best_state['epochs_trained'],
        'real': baseline, 'last': last, 'validation_selected': selected,
        'this_process_seconds': time.monotonic()-started, 'paper_reproduction': False})
    print((out/'final_report.json').read_text(), flush=True)


if __name__ == '__main__': main()
