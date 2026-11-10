"""One circuit definition shared by training, shot sampling and resource reports."""
import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumGeneratorCore(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, shots=1024,
                 diff_method="backprop", seed=42, hadamards_per_layer=False, finite_shot_training=False, ansatz="paper"):
        super().__init__()
        if n_qubits < 2 or n_layers < 1 or shots < 1:
            raise ValueError("Require at least two qubits, one layer and one shot")
        if ansatz not in {"paper", "reuploading", "paper_readout"}:
            raise ValueError("ansatz must be paper, reuploading or paper_readout")
        if ansatz == "reuploading" and (n_qubits % 2 or hadamards_per_layer):
            raise ValueError("reuploading requires even qubits and no repeated Hadamards")
        self.ansatz = ansatz
        self.n_qubits, self.n_layers, self.shots = n_qubits, n_layers, shots
        if finite_shot_training and diff_method != "parameter-shift":
            raise ValueError("Finite-shot training requires parameter-shift gradients")
        self.finite_shot_training = finite_shot_training
        self.diff_method = diff_method
        self.hadamards_per_layer = hadamards_per_layer
        self.weights = nn.Parameter(2 * np.pi * torch.rand(n_layers, n_qubits, 2))
        self.device = qml.device("default.qubit", wires=n_qubits, seed=seed)
        self.circuit = qml.QNode(self._probabilities, self.device, interface="torch",
                                 diff_method=diff_method)
        if finite_shot_training:
            self.circuit = qml.set_shots(self.circuit, shots=shots)
        sampling_device = qml.device("default.qubit", wires=n_qubits, seed=seed)
        self.sampling_circuit = qml.set_shots(
            qml.QNode(self._counts, sampling_device), shots=shots)

    def operations(self, noise, weights):
        if self.ansatz == "reuploading":
            # Experimental ansatz: phase rotations precede amplitude mixing, so
            # final-layer RZ parameters affect measured probabilities. Noise is
            # re-uploaded each layer. Two disjoint matchings replace a serial ring.
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    qml.RY(noise[..., wire], wires=wire)
                    qml.RZ(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                for offset in (0, 1):
                    for wire in range(offset, self.n_qubits, 2):
                        pair = [wire, (wire + 1) % self.n_qubits]
                        qml.CNOT(wires=pair if layer % 2 == 0 else pair[::-1])
            return
        # Paper leaves noise encoding and repeated Hadamards ambiguous. We use
        # one H initialization by default, RY noise, then L RY/RZ + rings.
        # Repeated layer Hadamards are an explicit alternative interpretation.
        for wire in range(self.n_qubits):
            qml.Hadamard(wires=wire)
            qml.RY(noise[..., wire], wires=wire)
        for layer in range(self.n_layers):
            if self.hadamards_per_layer and layer > 0:
                for wire in range(self.n_qubits):
                    qml.Hadamard(wires=wire)
            for wire in range(self.n_qubits):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            for wire in range(self.n_qubits):
                qml.CNOT(wires=[wire, (wire + 1) % self.n_qubits])
        if self.ansatz == "paper_readout":
            # Explicit extension: X-basis readout makes final RZ phases visible.
            # All paper variational blocks and the ordered CNOT ring are retained.
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)

    def _probabilities(self, noise, weights):
        self.operations(noise, weights)
        return qml.probs(wires=range(self.n_qubits))

    def _counts(self, noise, weights):
        self.operations(noise, weights)
        return qml.counts(wires=range(self.n_qubits))

    def forward(self, noise):
        if noise.ndim != 2 or noise.shape[1] != self.n_qubits:
            raise ValueError("noise must have shape (batch, n_qubits)")
        if self.diff_method == "parameter-shift":
            return torch.stack([self.circuit(row, self.weights) for row in noise]).float()
        return self.circuit(noise, self.weights).float()

    def sample_bitstrings(self, noise):
        weights = self.weights.detach().cpu().numpy()
        return [self.sampling_circuit(row.detach().cpu().numpy(), weights) for row in noise]

    def sampled_probabilities(self, noise):
        counts = self.sample_bitstrings(noise)
        return torch.tensor([[row.get(format(i, f"0{self.n_qubits}b"), 0) / self.shots
                              for i in range(2 ** self.n_qubits)] for row in counts],
                            dtype=torch.float32, device=noise.device)
