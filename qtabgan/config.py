from dataclasses import dataclass
from typing import Tuple


@dataclass
class QTabGANConfig:
    """
    Configuration for the QTabGAN model.

    The defaults are implementation choices based on the
    architecture described in the paper.
    """

    # Quantum circuit
    n_qubits: int = 8
    n_layers: int = 2
    shots: int = 1024
    diff_method: str = "backprop"
    hadamards_per_layer: bool = False
    finite_shot_training: bool = False

    # Classical mapper
    mapper_hidden: Tuple[int, ...] = (128, 128)

    # Discriminator
    discriminator_hidden: Tuple[int, ...] = (128, 64, 32)

    # Training
    batch_size: int = 32
    epochs: int = 30

    # Optimizers
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4

    # Device
    device: str = "cpu"

    # Reproducibility
    seed: int = 42
