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
    n_qubits: int = 4
    n_layers: int = 2
    shots: int = 256

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