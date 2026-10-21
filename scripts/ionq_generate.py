"""Prepare, locally test, submit, and collect IonQ checkpoint-generation jobs."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import requests
import torch
from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.ionq import (IonQClient, attach_job, bundle_lock, collect_bundle, digest,
                         export_qis, load_bundle, simulate_qis, submit_bundle, write_json)


def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    config = QTabGANConfig(**checkpoint["config"])
    # Hardware inference needs fixed weights, not a training estimator.
    config.device = "cpu"
    config.diff_method = "backprop"
    config.finite_shot_training = False
    state = checkpoint["state_dict"]
    data_dim = state["discriminator.model.network.0.weight"].shape[1]
    condition_dim = state["mapper.model.network.0.weight"].shape[1] - 2 ** config.n_qubits
    model = QTabGAN(data_dim, condition_dim, config)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f"Output bundle already exists: {args.output}")
    if args.rows < 1 or not 1 <= args.shots <= 1_000_000:
        raise ValueError("rows must be positive; shots must be between 1 and 1,000,000")
    if args.debias and args.shots < 500:
        raise ValueError("IonQ debiasing requires at least 500 shots")
    if args.backend != "simulator" and not args.backend.startswith("qpu."):
        raise ValueError("Choose simulator or an exact qpu.* backend from the backends command")
    root = args.run / f"seed_{args.seed}"
    checkpoint_path = root / "qtabgan/model.pt"
    preprocessor_path = root / "preprocessor.joblib"
    manifest = json.loads((args.run / "manifest.json").read_text())
    model, checkpoint = load_checkpoint(checkpoint_path)
    if not manifest["arguments"].get("data") or not manifest["arguments"].get("target"):
        raise ValueError("Use a CSV-backed classification benchmark with saved labels and target")
    labels = manifest["labels"]
    if len(labels) != model.condition_dim:
        raise ValueError("Checkpoint condition width differs from the saved label order")
    preprocessor = joblib.load(preprocessor_path)  # Trusted local benchmark artifact.
    if len(preprocessor.output_columns) != model.data_dim:
        raise ValueError("Preprocessor output width differs from the checkpoint")
    indices = np.array(args.class_indices if args.class_indices is not None
                       else [i % len(labels) for i in range(args.rows)], dtype=int)
    if len(indices) != args.rows or np.any(indices < 0) or np.any(indices >= len(labels)):
        raise ValueError("Supply exactly --rows class indices, each in the saved label range")
    conditions = np.eye(len(labels), dtype=np.float32)[indices]
    noise = np.random.default_rng(args.latent_seed).uniform(0, np.pi,
                    (args.rows, model.config.n_qubits)).astype(np.float32)
    circuits = [export_qis(model.quantum_generator, row) for row in noise]
    ideal = np.stack([simulate_qis(circuit) for circuit in circuits])
    with torch.no_grad():
        expected = model.quantum_generator(torch.from_numpy(noise)).numpy()
    export_error = float(np.max(np.abs(ideal - expected)))
    if export_error > 1e-6:
        raise AssertionError(f"Exported circuit does not match checkpoint: {export_error}")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(checkpoint_path, args.output / "checkpoint.pt")
    shutil.copy2(preprocessor_path, args.output / "preprocessor.joblib")
    bundle_id = uuid.uuid4().hex
    requests = [{"type": "ionq.circuit.v1", "name": f"qtabgan-{bundle_id}-row-{i}",
                 "backend": args.backend, "shots": args.shots,
                 "settings": {"error_mitigation": {"debiasing": args.debias}},
                 "metadata": {"qtabgan_bundle": bundle_id, "qtabgan_row": str(i)},
                 "input": circuit} for i, circuit in enumerate(circuits)]
    plan = {"schema_version": 1, "bundle_id": bundle_id, "provider": "ionq-direct-v0.4",
            "backend": args.backend, "rows": args.rows, "shots_per_circuit": args.shots,
            "requested_circuit_shots": args.rows * args.shots,
            "n_qubits": model.config.n_qubits, "config": asdict(model.config),
            "training_config": checkpoint["config"],
            "epochs_trained": checkpoint.get("epochs_trained"),
            "labels": labels, "target": manifest["arguments"]["target"],
            "class_indices": indices.tolist(), "conditions": conditions.tolist(),
            "noise": noise.tolist(), "ideal_probabilities": ideal.tolist(),
            "export_max_probability_error": export_error,
            "latent_seed": args.latent_seed, "requests": requests,
            "assets": {name: digest(args.output / name)
                       for name in ["checkpoint.pt", "preprocessor.joblib"]},
            "versions": {name: importlib.metadata.version(name)
                         for name in ["torch", "pennylane", "numpy", "scikit-learn", "requests"]}}
    write_json(args.output / "bundle.json", plan)
    write_json(args.output / "state.json", {"bundle_sha256": digest(args.output / "bundle.json"),
               "jobs": [{"status": "pending", "job_id": None} for _ in requests]})
    print(f"Prepared {args.rows} circuits, {model.config.n_qubits} qubits, "
          f"{args.shots} shots each ({args.rows * args.shots} requested circuit-shots), backend={args.backend}.")
    print(f"Export check: max probability error {export_error:.3g}. No cloud requests made.")
    print(f"Review {args.output / 'bundle.json'}; then use local or submit.")


def map_probabilities(directory, plan, probabilities, name):
    """Use the frozen classical mapper and saved inverse transform on this CPU."""
    directory = Path(directory)
    model, _ = load_checkpoint(directory / "checkpoint.pt")
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (plan["rows"], 2 ** plan["n_qubits"]):
        raise ValueError("Probability matrix has the wrong shape")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1.):
        raise ValueError("Invalid probabilities supplied to the mapper")
    with torch.no_grad():
        conditions = torch.tensor(plan["conditions"], dtype=torch.float32)
        features = model.mapper(torch.tensor(probabilities, dtype=torch.float32), conditions).numpy()
        ideal_features = model.mapper(torch.tensor(plan["ideal_probabilities"], dtype=torch.float32), conditions).numpy()
    frame = preprocessor.inverse_transform(features)
    frame[plan["target"]] = np.asarray(plan["labels"])[plan["class_indices"]]
    frame.to_csv(directory / f"{name}_synthetic.csv", index=False)
    np.savez(directory / f"{name}_probabilities.npz", probabilities=probabilities, features=features)
    ideal = np.array(plan["ideal_probabilities"])
    write_json(directory / f"{name}_diagnostics.json", {
        "backend": "local" if name.startswith("local") else plan["backend"],
        "rows": plan["rows"], "shots_requested_per_circuit": plan["shots_per_circuit"],
        "mean_probability_total_variation_from_ideal": float(np.abs(probabilities - ideal).sum(1).mean()/2),
        "mean_absolute_mapper_feature_shift_from_ideal": float(np.abs(features - ideal_features).mean()),
        "checkpoint_quality": "Existing checkpoint; hardware execution does not establish utility or paper reproduction",
        "scope": "Frozen checkpoint generation, not hardware training"})
    print(f"Wrote {directory / (name + '_synthetic.csv')}")


def local(directory):
    directory = Path(directory)
    with bundle_lock(directory):
        plan, _ = load_bundle(directory)
        # Replay the exact JSON that submit will send, rather than just the source QNode.
        ideal = np.stack([simulate_qis(request["input"]) for request in plan["requests"]])
        np.testing.assert_allclose(ideal, plan["ideal_probabilities"], atol=1e-10, rtol=1e-8)
        rng = np.random.default_rng(plan["latent_seed"] + 1)
        shots = plan["shots_per_circuit"]
        sampled = np.stack([rng.multinomial(shots, row) / shots for row in ideal])
        map_probabilities(directory, plan, ideal, "local_ideal")
        map_probabilities(directory, plan, sampled, "local_shots")
    print("Local validation complete. No IonQ account or network used.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Create an offline, reviewable execution bundle")
    p.add_argument("--run", type=Path, required=True, help="Completed CSV-backed benchmark directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--class-indices", type=int, nargs="+")
    p.add_argument("--latent-seed", type=int, default=1234)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--backend", default="simulator")
    p.add_argument("--debias", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    for name in ["local", "submit", "collect"]:
        p = commands.add_parser(name)
        p.add_argument("bundle", type=Path)
    p = commands.add_parser("attach", help="Recover a job ID after an uncertain submission")
    p.add_argument("bundle", type=Path)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--job-id", required=True)
    commands.add_parser("backends", help="Read the backends visible to your IonQ API key")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "prepare": prepare(args)
        elif args.command == "local": local(args.bundle)
        elif args.command == "backends": print(json.dumps(IonQClient().backends(), indent=2))
        elif args.command == "submit": submit_bundle(args.bundle, IonQClient())
        elif args.command == "attach": attach_job(args.bundle, args.row, args.job_id, IonQClient())
        elif args.command == "collect":
            probabilities = collect_bundle(args.bundle, IonQClient())
            if probabilities is not None:
                with bundle_lock(args.bundle):
                    plan, _ = load_bundle(args.bundle)
                    map_probabilities(args.bundle, plan, probabilities, "ionq")
            else:
                _, state = load_bundle(args.bundle)
                failed = [i for i, job in enumerate(state["jobs"])
                          if job["status"] in {"failed", "canceled", "cancelled"}]
                if failed:
                    parser.exit(1, f"IonQ jobs failed/canceled for rows {failed}; inspect row job files. No complete table written.\n")
                print("Not all rows completed. Rerun collect later; it never submits jobs.")
    except requests.RequestException:
        parser.exit(1, "IonQ network request failed. Inspect state.json before retrying a submission; an accepted job may need attach.\n")
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
