"""Reproducible classification comparison; see README for protocol and caveats."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import joblib
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score, precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.preprocessing import TabularPreprocessor
from qtabgan.evaluation import average_categorical_jsd, correlation_difference
from qtabgan.utils import set_seed


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate(x, y, test, truth, out, labels, seed):
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(x, y)
    pred = clf.predict(test)
    result = {"accuracy": float(accuracy_score(truth, pred)),
              "f1_macro": float(f1_score(truth, pred, average="macro", zero_division=0)),
              "f1_weighted": float(f1_score(truth, pred, average="weighted", zero_division=0)),
              "confusion_matrix": confusion_matrix(truth, pred, labels=np.arange(len(labels))).tolist()}
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=labels).plot(ax=ax)
    fig.tight_layout(); fig.savefig(out / "confusion.png"); plt.close(fig)
    if len(labels) == 2 and len(clf.classes_) == 2:
        prob = clf.predict_proba(test)[:, list(clf.classes_).index(1)]
        fpr, tpr, _ = roc_curve(truth, prob)
        precision, recall, _ = precision_recall_curve(truth, prob)
        result.update(roc_auc=float(roc_auc_score(truth, prob)),
                      average_precision=float(average_precision_score(truth, prob)),
                      f1_binary=float(f1_score(truth, pred, zero_division=0)))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label=f'AUROC={result["roc_auc"]:.3f}')
        axes[0].plot([0, 1], [0, 1], "--"); axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
        axes[1].plot(recall, precision, label=f'AP={result["average_precision"]:.3f}')
        axes[1].axhline(np.mean(truth == 1), linestyle="--")
        axes[1].set(xlabel="Recall", ylabel="Precision")
        for ax in axes: ax.legend()
        fig.tight_layout(); fig.savefig(out / "auc_curves.png"); plt.close(fig)
    save_json(out / "utility.json", result)
    return result


def distribution(real, fake, columns, out):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for j, name in enumerate(columns):
        a, b = real[:, j], fake[:, j]
        rows.append({"feature": str(name), "ks": float(ks_2samp(a, b, method="asymp").statistic),
                     "wasserstein": float(wasserstein_distance(a, b)),
                     "synthetic_std": float(b.std()), "real_std": float(a.std())})
        if j < 6:
            ax = axes.flat[j]
            for values, label in [(a, "real train"), (b, "synthetic")]:
                ax.plot(np.sort(values), np.arange(1, len(values)+1)/len(values), label=label)
            ax.set_title(str(name)); ax.legend()
    fig.tight_layout(); fig.savefig(out / "cdf.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "feature_diagnostics.csv", index=False)
    # Constant generated columns are flagged separately; omit undefined correlations.
    valid = (real.std(0) > 1e-10) & (fake.std(0) > 1e-10)
    corr = None
    if valid.sum() >= 2:
        corr = float(np.abs(np.corrcoef(real[:, valid], rowvar=False) -
                            np.corrcoef(fake[:, valid], rowvar=False)).mean())
    return {"mean_ks": float(np.mean([r["ks"] for r in rows])),
            "mean_wasserstein": float(np.mean([r["wasserstein"] for r in rows])),
            "mean_absolute_correlation_difference": corr,
            "unique_pair_correlation_difference": correlation_difference(real, fake),
            "constant_synthetic_columns": int((fake.std(0) <= 1e-10).sum())}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path)
    p.add_argument("--target")
    p.add_argument("--categorical", nargs="*", default=[])
    p.add_argument("--dataset-name", default="smoke-breast-cancer")
    p.add_argument("--models", nargs="+", choices=["qtabgan", "ctgan"], default=["qtabgan", "ctgan"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--checkpoints", type=int, nargs="*", default=[])
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hadamards-per-layer", action="store_true")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--finite-shot-training", action="store_true")
    p.add_argument("--finite-shot-evaluation", action="store_true")
    p.add_argument("--diff-method", choices=["backprop", "parameter-shift"], default="backprop")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--test-size", type=float, default=.2)
    p.add_argument("--max-rows", type=int)
    p.add_argument("--output", type=Path, default=Path("results/benchmark"))
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.batch_size % 2:
        p.error("epochs must be positive; batch size must be positive and even")
    if args.finite_shot_training and args.diff_method != "parameter-shift":
        p.error("--finite-shot-training requires --diff-method parameter-shift")
    if args.data and not args.target: p.error("--data requires --target")
    torch.set_num_threads(1)
    if args.data:
        df = pd.read_csv(args.data)
        source_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
        target = args.target
    else:
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame; target = "target"; source_hash = "sklearn.load_breast_cancer"
    if df.isna().any().any(): p.error("Missing values: clean explicitly before benchmarking")
    cat = args.categorical
    numerical = [c for c in df.columns if c != target and c not in cat]
    y, labels = pd.factorize(df[target], sort=True)
    features = df.drop(columns=target)
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {k: importlib.metadata.version(k) for k in ["torch", "pennylane", "numpy", "pandas", "scikit-learn", "ctgan"]}
    source_root = Path(__file__).resolve().parent
    source_files = [source_root / "benchmark.py", *sorted((source_root / "qtabgan").glob("*.py"))]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    save_json(args.output / "manifest.json", {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              "versions": versions, "source_files_sha256": source_hashes, "source_sha256": source_hash, "labels": [str(v) for v in labels],
              "paper_reproduction": False, "f1_convention": "macro, weighted and binary reported; paper unspecified"})
    summaries = []
    for seed in args.seeds:
        set_seed(seed)
        indices = np.arange(len(df))
        if args.max_rows and args.max_rows < len(df):
            indices, _ = train_test_split(indices, train_size=args.max_rows, stratify=y, random_state=seed)
        tr, te = train_test_split(indices, test_size=args.test_size, stratify=y[indices], random_state=seed)
        root = args.output / f"seed_{seed}"; root.mkdir(exist_ok=True)
        np.savez(root / "split.npz", train=tr, test=te)
        pre = TabularPreprocessor(numerical, cat, noise_std=0)
        x = pre.fit_transform(features.iloc[tr]); test = pre.transform(features.iloc[te])
        joblib.dump(pre, root / "preprocessor.joblib")
        base_dir = root / "real"; base_dir.mkdir(exist_ok=True)
        base = evaluate(x, y[tr], test, y[te], base_dir, labels, seed)
        cond = np.eye(len(labels), dtype=np.float32)[y[tr]]
        for name in args.models:
            set_seed(seed)
            out = root / name; out.mkdir(exist_ok=True)
            start = time.perf_counter()
            if name == "qtabgan":
                cfg = QTabGANConfig(n_qubits=args.qubits, n_layers=args.layers, shots=args.shots,
                    diff_method=args.diff_method, hadamards_per_layer=args.hadamards_per_layer, finite_shot_training=args.finite_shot_training, epochs=args.epochs, batch_size=args.batch_size, seed=seed)
                model = QTabGAN(x.shape[1], len(labels), cfg)
                all_history = {}; previous = 0
                checkpoints = sorted(set([c for c in args.checkpoints if 0 < c < args.epochs] + [args.epochs]))
                for epoch in checkpoints:
                    # Keep Adam state across evaluation checkpoints.
                    model.config.epochs = epoch - previous
                    history = model.fit(x, cond)
                    for k,v in history.items(): all_history.setdefault(k, []).extend(v)
                    checkpoint_out = out / f"epoch_{epoch}"; checkpoint_out.mkdir(exist_ok=True)
                    generated = model.generate(cond, sample=args.finite_shot_evaluation)
                    decoded = pre.inverse_transform(generated)
                    fake = pre.transform(decoded)
                    utility = evaluate(fake, y[tr], test, y[te], checkpoint_out, labels, seed)
                    fidelity = distribution(x, fake, pre.output_columns, checkpoint_out)
                    save_json(checkpoint_out / "metrics.json", {"utility": utility, "fidelity": fidelity})
                    previous = epoch
                pd.DataFrame(all_history).to_csv(out / "losses.csv", index=False)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for k in ["generator_loss", "discriminator_loss"]: axes[0].plot(np.arange(1,args.epochs+1), all_history[k], label=k)
                axes[0].legend(); axes[0].set_xlabel("Epoch")
                axes[1].plot(np.arange(1,args.epochs+1), all_history["quantum_gradient_norm"])
                axes[1].set(xlabel="Epoch", ylabel="Quantum gradient norm")
                fig.tight_layout(); fig.savefig(out / "losses.png"); plt.close(fig)
                cfg.epochs = args.epochs
                torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epochs_trained":args.epochs}, out / "model.pt")
                core = model.quantum_generator
                z = torch.zeros(args.qubits)
                (out / "circuit.txt").write_text(qml.draw(core.circuit)(z, core.weights))
                fig, _ = qml.draw_mpl(core.circuit)(z, core.weights)
                fig.savefig(out / "circuit.png", bbox_inches="tight"); plt.close(fig)
                resources = qml.specs(core.circuit)(z, core.weights)["resources"]
                save_json(out / "circuit_resources.json", {"resources": str(resources), "trainable_angles":core.weights.numel(),
                    "cnot_count":args.qubits * args.layers, "optimization_applied":False})
            else:
                from ctgan import CTGAN
                table = features.iloc[tr].copy(); table[target] = y[tr]
                model = CTGAN(epochs=args.epochs, batch_size=args.batch_size, pac=2, enable_gpu=False,
                              generator_dim=(128,128), discriminator_dim=(128,64,32), verbose=True)
                model.set_random_state(seed); model.fit(table, discrete_columns=cat+[target])
                decoded_all = model.sample(len(tr)); decoded = decoded_all.drop(columns=target)
                fake = pre.transform(decoded); synthetic_y = decoded_all[target].to_numpy(dtype=int)
                model.loss_values.to_csv(out / "losses.csv", index=False)
                fig, ax = plt.subplots()
                for column in ["Generator Loss", "Discriminator Loss"]:
                    ax.plot(model.loss_values["Epoch"], model.loss_values[column], label=column)
                ax.legend(); fig.savefig(out / "losses.png"); plt.close(fig)
                model.save(str(out / "model.pkl"))
            synthetic_y = y[tr] if name == "qtabgan" else synthetic_y
            decoded[target] = np.asarray(labels)[synthetic_y]
            decoded.to_csv(out / "synthetic.csv", index=False)
            utility = evaluate(fake, synthetic_y, test, y[te], out, labels, seed)
            fidelity = distribution(x, fake, pre.output_columns, out)
            # Decode before JSD: continuous mapper outputs are not categorical codes.
            fidelity["categorical_jsd"] = None
            if cat:
                a = features.iloc[tr][cat].astype(str).to_numpy(); b = decoded[cat].astype(str).to_numpy()
                fidelity["categorical_jsd"] = average_categorical_jsd(a, b, range(len(cat)))
            summary = {"model":name, "seed":seed, "epochs":args.epochs, "seconds":time.perf_counter()-start,
                **utility, **fidelity, "accuracy_gap":abs(base["accuracy"]-utility["accuracy"]),
                "f1_macro_gap":abs(base["f1_macro"]-utility["f1_macro"])}
            save_json(out / "metrics.json", summary); summaries.append(summary)
            pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    numeric = pd.DataFrame(summaries).select_dtypes(include="number").columns
    pd.DataFrame(summaries).groupby("model")[numeric].agg(["mean", "std"]).to_csv(args.output / "aggregate.csv")


if __name__ == "__main__": main()
