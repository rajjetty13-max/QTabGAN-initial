"""IonQ QIS export and explicit, resumable checkpoint inference.

REST contract: https://docs.ionq.com/guides/direct-api-submission
No cloud requests occur while importing, preparing, or locally simulating a bundle.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pennylane as qml
import torch

API_URL = "https://api.ionq.co/v0.4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Replace a journal atomically, including after each accepted cloud job."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def bundle_lock(directory):
    with (Path(directory) / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another command is using this bundle") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def export_qis(core, noise):
    """Export the existing PennyLane gate definition; QIS rotations use radians."""
    noise = torch.as_tensor(noise, dtype=core.weights.dtype, device=core.weights.device)
    if noise.shape != (core.n_qubits,) or not torch.isfinite(noise).all():
        raise ValueError("One finite noise angle per qubit is required")
    tape = qml.tape.make_qscript(core._probabilities)(noise, core.weights.detach())
    gates = []
    for operation in tape.operations:
        wires = [int(wire) for wire in operation.wires]
        if operation.name == "Hadamard":
            gate = {"gate": "h", "target": wires[0]}
        elif operation.name in {"RY", "RZ"}:
            angle = float(operation.parameters[0])
            if not np.isfinite(angle):
                raise ValueError("Nonfinite quantum rotation")
            gate = {"gate": operation.name.lower(), "target": wires[0], "rotation": angle}
        elif operation.name == "CNOT":
            gate = {"gate": "cnot", "control": wires[0], "target": wires[1]}
        else:
            raise ValueError(f"Unsupported IonQ export operation: {operation.name}")
        gates.append(gate)
    return {"qubits": core.n_qubits, "gateset": "qis", "circuit": gates}


def simulate_qis(circuit):
    """Replay the exported JSON locally, independently of the model's QNode."""
    n = circuit["qubits"]
    if circuit.get("gateset") != "qis":
        raise ValueError("Only QIS circuits are supported")
    @qml.qnode(qml.device("default.qubit", wires=n))
    def execute():
        for gate in circuit["circuit"]:
            name = gate["gate"]
            if name == "h": qml.Hadamard(gate["target"])
            elif name == "ry": qml.RY(gate["rotation"], gate["target"])
            elif name == "rz": qml.RZ(gate["rotation"], gate["target"])
            elif name == "cnot": qml.CNOT([gate["control"], gate["target"]])
            else: raise ValueError(f"Unsupported QIS gate: {name}")
        return qml.probs(wires=range(n))
    return np.asarray(execute(), dtype=float)


def probabilities_from_ionq(histogram, n_qubits):
    """Sparse decimal keys use q_i at bit 2**i; mapper puts wire 0 first/MSB.

    Accept the flat QIS probability distribution documented by the direct API.
    Do not guess ordering for other registers, bitstring formats, or child jobs.
    """
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError("Expected a nonempty flat IonQ QIS probability object")
    probabilities = np.zeros(2 ** n_qubits, dtype=float)
    for key, value in histogram.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
            raise ValueError("IonQ outcomes must be decimal integer strings")
        outcome = int(key)
        if str(outcome) != key or not 0 <= outcome < len(probabilities):
            raise ValueError("Invalid or out-of-range IonQ outcome")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("IonQ probabilities must be numbers")
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid IonQ probability")
        mapper_index = int(format(outcome, f"0{n_qubits}b")[::-1], 2)
        probabilities[mapper_index] = value
    if not np.isclose(probabilities.sum(), 1., atol=1e-5, rtol=1e-5):
        raise ValueError("IonQ probabilities must sum to one; counts are not probabilities")
    return probabilities / probabilities.sum()


class IonQClient:
    """Small v0.4 QIS client. POST requests are never automatically retried."""
    def __init__(self, api_key=None, session=None):
        import requests
        self._key = api_key or os.environ.get("IONQ_API_KEY")
        if not self._key:
            raise ValueError("Set IONQ_API_KEY in your local environment")
        self._session = session or requests.Session()

    def _request(self, method, path, payload=None):
        response = self._session.request(
            method, API_URL + path,
            headers={"Authorization": "apiKey " + self._key},
            json=payload, timeout=30, allow_redirects=False)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"IonQ API returned HTTP {response.status_code} for {method} {path}")
        return response.json()

    def backends(self):
        return self._request("GET", "/backends")

    def submit(self, payload):
        return self._request("POST", "/jobs", payload)

    def job(self, job_id):
        job_id = str(uuid.UUID(job_id))
        return self._request("GET", f"/jobs/{job_id}")

    def probabilities(self, job_id, job):
        job_id = str(uuid.UUID(job_id))
        # Prefer the unambiguous v1 decimal artifact when the job provides one.
        artifact = job.get("results", {}).get("ionq.result.probabilities.json.v1")
        if artifact:
            artifact_id = str(uuid.UUID(artifact["id"]))
            return self._request("GET", f"/jobs/{job_id}/artifacts/{artifact_id}")
        # The documented QIS endpoint returns a flat decimal-key distribution,
        # including on deployments which expose newer artifact descriptors.
        return self._request("GET", f"/jobs/{job_id}/results/probabilities")


def load_bundle(directory):
    directory = Path(directory)
    plan = json.loads((directory / "bundle.json").read_text())
    state = json.loads((directory / "state.json").read_text())
    if plan.get("schema_version") != 1 or state["bundle_sha256"] != digest(directory / "bundle.json"):
        raise ValueError("Bundle plan changed or has an unsupported schema")
    for name, expected in plan["assets"].items():
        if name not in {"checkpoint.pt", "preprocessor.joblib"}:
            raise ValueError("Unexpected bundle asset")
        if digest(directory / name) != expected:
            raise ValueError(f"Bundle asset changed: {name}")
    if len(state["jobs"]) != len(plan["requests"]):
        raise ValueError("Bundle job journal has the wrong number of rows")
    return plan, state


def submit_bundle(directory, client):
    """Save 'submitting' before POST, so a lost response cannot trigger a repeat."""
    directory = Path(directory)
    with bundle_lock(directory):
        plan, state = load_bundle(directory)
        for index, (payload, entry) in enumerate(zip(plan["requests"], state["jobs"])):
            if entry.get("job_id"):
                continue
            if entry["status"] != "pending":
                raise RuntimeError(f"Row {index} has an uncertain submission. Recover its job ID with attach; do not resubmit blindly.")
            entry["status"] = "submitting"
            write_json(directory / "state.json", state)
            # If interrupted or timed out, the durable state remains submitting.
            result = client.submit(payload)
            job_id = str(uuid.UUID(result["id"]))
            entry.update(job_id=job_id, status="submitted")
            write_json(directory / "state.json", state)
            print(f"Row {index}: submitted {job_id}", flush=True)
        return state


def attach_job(directory, row, job_id, client):
    """Recover an accepted job after a lost POST response, without another POST."""
    directory = Path(directory)
    with bundle_lock(directory):
        plan, state = load_bundle(directory)
        if not 0 <= row < len(state["jobs"]):
            raise ValueError("Row index is out of range")
        entry = state["jobs"][row]
        if entry.get("job_id"):
            raise ValueError("Row already has a job ID")
        job_id = str(uuid.UUID(job_id))
        job = client.job(job_id)
        request = plan["requests"][row]
        if job.get("name") != request["name"] or job.get("backend") != request["backend"]:
            raise ValueError("Job name/backend does not match this bundle row")
        entry.update(job_id=job_id, status="submitted")
        write_json(directory / "state.json", state)


def collect_bundle(directory, client):
    """One status check per unfinished row. Never submits or waits in a loop."""
    directory = Path(directory)
    with bundle_lock(directory):
        plan, state = load_bundle(directory)
        for index, entry in enumerate(state["jobs"]):
            if entry["status"] == "completed":
                continue
            job_id = entry.get("job_id")
            if not job_id:
                print(f"Row {index}: {entry['status']} (no job ID)")
                continue
            job = client.job(job_id)
            request = plan["requests"][index]
            if job.get("backend") != request["backend"] or job.get("name") != request["name"]:
                raise ValueError("Returned job does not match the submitted bundle")
            write_json(directory / f"row_{index:04d}_job.json", job)
            status = job["status"]
            if status == "completed":
                raw = client.probabilities(job_id, job)
                probabilities = probabilities_from_ionq(raw, plan["n_qubits"])
                write_json(directory / f"row_{index:04d}_result.json", raw)
                entry.update(status="completed", probabilities=probabilities.tolist())
            elif status in {"failed", "canceled", "cancelled"}:
                entry["status"] = status
            elif status in {"submitted", "ready", "started", "running", "queued"}:
                entry["status"] = status
            else:
                raise ValueError(f"Unrecognized IonQ job status: {status}")
            write_json(directory / "state.json", state)
            print(f"Row {index}: {entry['status']}", flush=True)
        if all(entry["status"] == "completed" for entry in state["jobs"]):
            return np.array([entry["probabilities"] for entry in state["jobs"]], dtype=float)
        return None
