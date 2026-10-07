import numpy as np
import pennylane as qml
import torch
from torch import nn


class QuantumGeneratorCore(nn.Module):
    """
    Variational quantum circuit used by QTabGAN.

    The circuit contains:
        - Hadamard initialization
        - RY/RZ trainable rotations
        - Circular CNOT entanglement
        - Probability-vector output
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        shots=256,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        # Trainable quantum parameters
        self.weights = nn.Parameter(
            0.01 * torch.randn(
                n_layers,
                n_qubits,
                2
            )
        )

        @qml.qnode(
            self.device,
            interface="torch",
            diff_method="parameter-shift"
        )
        def circuit(noise, weights):

            # Hadamard initialization
            for qubit in range(n_qubits):
                qml.Hadamard(
                    wires=qubit
                )

            # Encode random noise
            for qubit in range(n_qubits):

                qml.RY(
                    noise[qubit],
                    wires=qubit
                )

            # Trainable layers
            for layer in range(n_layers):

                for qubit in range(n_qubits):

                    qml.RY(
                        weights[layer, qubit, 0],
                        wires=qubit
                    )

                    qml.RZ(
                        weights[layer, qubit, 1],
                        wires=qubit
                    )

                # Circular CNOT entanglement
                for qubit in range(n_qubits):

                    target = (
                        qubit + 1
                    ) % n_qubits

                    qml.CNOT(
                        wires=[
                            qubit,
                            target
                        ]
                    )

            return qml.probs(
                wires=range(n_qubits)
            )

        self.circuit = circuit

    def forward(self, noise):
        """
        noise shape:
            [batch_size, n_qubits]

        Returns:
            [batch_size, 2**n_qubits]
        """

        outputs = []

        for sample in noise:

            probabilities = self.circuit(
                sample,
                self.weights
            )

            outputs.append(probabilities)

        return torch.stack(outputs).float()

    def sample_bitstrings(self, noise):
        """
        Generate actual finite-shot samples.

        This is intended for final sampling/evaluation rather
        than differentiable GAN training.
        """

        sampling_device = qml.device(
            "default.qubit",
            wires=self.n_qubits,
            shots=self.shots
        )

        @qml.qnode(sampling_device)
        def sampling_circuit(
            noise_sample,
            weights
        ):

            for qubit in range(self.n_qubits):

                qml.Hadamard(
                    wires=qubit
                )

                qml.RY(
                    noise_sample[qubit],
                    wires=qubit
                )

            for layer in range(self.n_layers):

                for qubit in range(self.n_qubits):

                    qml.RY(
                        weights[
                            layer,
                            qubit,
                            0
                        ],
                        wires=qubit
                    )

                    qml.RZ(
                        weights[
                            layer,
                            qubit,
                            1
                        ],
                        wires=qubit
                    )

                for qubit in range(self.n_qubits):

                    target = (
                        qubit + 1
                    ) % self.n_qubits

                    qml.CNOT(
                        wires=[
                            qubit,
                            target
                        ]
                    )

            return qml.counts(
                wires=range(self.n_qubits)
            )

        results = []

        weights = (
            self.weights
            .detach()
            .cpu()
            .numpy()
        )

        for sample in noise:

            counts = sampling_circuit(
                sample.detach().cpu().numpy(),
                weights
            )

            results.append(counts)

        return results