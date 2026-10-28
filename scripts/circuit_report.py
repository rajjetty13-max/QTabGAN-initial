"""Check an exact, gate-preserving simplification before hardware compilation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch
from qtabgan.quantum_generator import QuantumGeneratorCore

p = argparse.ArgumentParser(__doc__)
p.add_argument('--qubits', type=int, default=8)
p.add_argument('--layers', type=int, default=2)
p.add_argument('--hadamards-per-layer', action='store_true')
p.add_argument('--checkpoint', type=Path)
p.add_argument('--output', type=Path, default=Path('results/circuit'))
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
core = QuantumGeneratorCore(a.qubits, a.layers, hadamards_per_layer=a.hadamards_per_layer).double()
if a.checkpoint:
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    if cfg.get('ansatz', 'paper') != 'paper':
        p.error('This probability-preserving simplification applies only to the paper ansatz; use compare_circuits.py for the alternative')
    if (cfg['n_qubits'], cfg['n_layers']) != (a.qubits, a.layers):
        p.error('Checkpoint qubits/layers must match --qubits and --layers')
    if cfg.get('hadamards_per_layer', False) != a.hadamards_per_layer:
        p.error('Checkpoint Hadamard setting must match --hadamards-per-layer')
    core.load_state_dict({'weights':checkpoint['state_dict']['quantum_generator.weights']})
def measured_circuit(noise, weights):
    tape = qml.tape.make_qscript(core._probabilities)(noise, weights)
    # Final RZ gates only change phases before a computational-basis permutation.
    # Removing them preserves probabilities, but need not preserve the statevector.
    rz_positions = [i for i, gate in enumerate(tape.operations) if gate.name == 'RZ']
    removable = set(rz_positions[-a.qubits:])
    for i, gate in enumerate(tape.operations):
        if i not in removable:
            qml.apply(gate)
    return qml.probs(wires=range(a.qubits))
optimized = qml.transforms.merge_rotations(qml.QNode(measured_circuit, core.device))
noise = torch.rand(8, a.qubits, dtype=torch.float64) * np.pi
with torch.no_grad():
    error = float((core.circuit(noise, core.weights)-optimized(noise, core.weights)).abs().max())
if error > 1e-10: raise AssertionError(f'Circuit equivalence failed: {error}')
report = {'checkpoint':str(a.checkpoint) if a.checkpoint else None,
          'max_probability_error_8_inputs':error, 'qubits':a.qubits, 'layers':a.layers, 'hadamards_per_layer':a.hadamards_per_layer,
          'original_trainable_angles':2*a.qubits*a.layers,
          'retained_trainable_angles':a.qubits*(2*a.layers-1),
          'scope':'Merge adjacent RY and omit final RZ before basis permutation. Probability equivalence only; ordered CNOT ring preserved. Logical resources only.'}
for name, circuit in [('original', core.circuit), ('merged', optimized)]:
    z = noise[0]
    report[name] = str(qml.specs(circuit)(z, core.weights)['resources'])
    (a.output / f'{name}.txt').write_text(qml.draw(circuit, level='device')(z, core.weights))
    fig, _ = qml.draw_mpl(circuit, level='device')(z, core.weights)
    fig.savefig(a.output / f'{name}.png', bbox_inches='tight'); plt.close(fig)
(a.output / 'resources.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
