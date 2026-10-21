# Noise simulation integration pointers

Establish noiseless utility and fidelity first. These are integration notes, not results from a hardware noise experiment.

## Where to connect a simulator

`QuantumGeneratorCore.operations(noise, weights)` in `quantum_generator.py` emits the common H, RY, RZ and ordered CNOT ring. Use that definition for exports and noise tests. `forward` currently returns exact statevector probabilities; `sampled_probabilities` returns finite-shot histograms. Shot uncertainty is distinct from gate/readout noise.

For local exploratory channel noise, construct a separate PennyLane `default.mixed` QNode around the same operations and insert the desired channels immediately after their corresponding gates. Do not add one channel at the end and call it a device noise model. Keep the ideal reference unchanged, use the same trained weights/noise inputs/labels, and compare empirical histograms at matched shots. Use density-matrix backpropagation only for simulator experiments; use parameter-shift or another supported estimator for hardware-oriented gradients.

## IonQ

The direct-cloud checkpoint inference adapter is now implemented. See [IONQ_HARDWARE.md](../IONQ_HARDWARE.md) for prepare/local/submit/collect commands, API-key setup, output ordering, and job recovery. It exports `QuantumGeneratorCore.operations` through its PennyLane tape, using QIS radians. Existing hardware checkpoints have only been validated locally; no live QPU result is claimed. The remaining notes describe the subsequent noise-study phase.

Start with IonQ's [hardware noise model demo](https://www.ionq.com/resources/get-started-with-hardware-noise-model-simulation). Verify the account's currently supported simulator targets, device noise models, gate set and shot limits before submitting. No credentials or cloud calls are embedded here.

Export H/RY/RZ/CNOT using the public SDK appropriate for the chosen API version; preserve radians, wire order, angle order and the directed final CNOT. Request simulator execution with an explicitly selected hardware noise model, and retain the response's actual settings. IonQ native compilation uses a different entangling basis, so compare native gate counts after compilation rather than interpreting logical CNOT depth as hardware depth.

## IBM

IBM documents [backend-derived noise models](https://quantum.cloud.ibm.com/docs/en/guides/build-noise-models) and [local Aer simulation](https://quantum.cloud.ibm.com/docs/en/guides/local-simulators). Use a selected fake backend for offline experiments or a dated calibration snapshot:

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# `backend` is the chosen fake or calibrated backend.
# `qc` is an export of QuantumGeneratorCore.operations at fixed parameters.
simulator = AerSimulator.from_backend(backend)
compiled = transpile(qc, simulator, optimization_level=3, seed_transpiler=42)
counts = simulator.run(compiled, shots=1024, seed_simulator=42).result().get_counts()
```

Measure all wires before running. PennyLane probability ordering treats wire 0 as the most significant bit; Qiskit count strings conventionally put classical bit 0 on the right. With qubit i measured into classical bit i, reverse count strings before building this mapper's probability vector. Verify with asymmetric basis-state tests, not only a Bell state.

## What to measure next

1. Freeze a trained checkpoint, mapper and test protocol. Compare ideal analytic, ideal finite-shot, and noisy finite-shot generation separately.
2. Evaluate original and simplified circuits using identical inputs; repeat with independent shot seeds. Compile both to the chosen backend and record native two-qubit count, depth, layout and calibration date.
3. Report probability total-variation distance and synthetic-data utility/fidelity, including class-level confusion counts. Repeat over model seeds before drawing conclusions about robustness.
4. Only then consider retraining under noise, changing the layer count or native entangler, or submitting hardware jobs.

The last RZ layer has zero ideal probability sensitivity. Its removal is valid for ideal computational-basis probabilities; a noisy implementation can behave differently because gate errors depend on the operations executed. Reordering the directed CNOT ring is not a generally valid depth optimization.
