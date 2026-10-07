import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import QTabGANConfig
from .quantum_generator import QuantumGeneratorCore
from .networks import (
    ClassicalMapper,
    Discriminator,
)
from .utils import set_seed


class QTabGAN(nn.Module):
    """
    Hybrid quantum-classical GAN for tabular data.

    Architecture:

        random noise
              |
              v
        quantum circuit
              |
              v
        probability vector
              |
              +------ condition
              |
              v
       classical mapper
              |
              v
        synthetic table

        discriminator
        /           \\
      real         fake
    """

    def __init__(
        self,
        data_dim,
        condition_dim,
        config=None,
    ):
        super().__init__()

        if config is None:
            config = QTabGANConfig()

        self.config = config

        self.data_dim = data_dim
        self.condition_dim = condition_dim

        set_seed(config.seed)

        # Quantum generator
        self.quantum_generator = (
            QuantumGeneratorCore(
                n_qubits=config.n_qubits,
                n_layers=config.n_layers,
                shots=config.shots,
            )
        )

        quantum_dim = (
            2 ** config.n_qubits
        )

        # Classical mapper
        self.mapper = ClassicalMapper(
            quantum_dim=quantum_dim,
            condition_dim=condition_dim,
            output_dim=data_dim,
            hidden_dims=config.mapper_hidden
        )

        # Discriminator
        self.discriminator = Discriminator(
            input_dim=data_dim,
            hidden_dims=config.discriminator_hidden
        )

    def _noise(self, batch_size):

        return (
            torch.rand(
                batch_size,
                self.config.n_qubits,
                device=self.config.device
            )
            * np.pi
        )

    def generate_from_condition(
        self,
        condition
    ):
        """
        Generate synthetic samples using a supplied condition.
        """

        batch_size = condition.shape[0]

        noise = self._noise(
            batch_size
        )

        quantum_probability = (
            self.quantum_generator(
                noise
            )
        )

        synthetic = self.mapper(
            quantum_probability,
            condition
        )

        return synthetic

    def generate(
    self,
    condition,
    sample=True
    ):
        """
        Public generation function.

        Accepts either a NumPy array or PyTorch tensor.
        """

        self.eval()

        if not isinstance(condition, torch.Tensor):
            condition = torch.tensor(
                condition,
                dtype=torch.float32,
                device=self.config.device
            )
        else:
            condition = condition.to(
                device=self.config.device,
                dtype=torch.float32
            )

        with torch.no_grad():

            synthetic = (
                self.generate_from_condition(
                 condition
                )
            )

        return synthetic.cpu().numpy()

    def fit(
        self,
        data,
        conditions,
        verbose=True
    ):
        """
        Train QTabGAN.

        Parameters
        ----------
        data:
            Preprocessed real tabular data.

        conditions:
            Condition vector for each row.
        """

        data_tensor = torch.tensor(
            data,
            dtype=torch.float32,
            device=self.config.device
        )

        condition_tensor = torch.tensor(
            conditions,
            dtype=torch.float32,
            device=self.config.device
        )

        dataset = TensorDataset(
            data_tensor,
            condition_tensor
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        generator_parameters = list(
            self.quantum_generator.parameters()
        ) + list(
            self.mapper.parameters()
        )

        generator_optimizer = torch.optim.Adam(
            generator_parameters,
            lr=self.config.lr_generator
        )

        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr_discriminator
        )

        criterion = nn.BCEWithLogitsLoss()

        history = {
            "generator_loss": [],
            "discriminator_loss": []
        }

        for epoch in range(
            self.config.epochs
        ):

            generator_total = 0.0
            discriminator_total = 0.0

            batches = 0

            for real_data, condition in loader:

                batches += 1

                batch_size = (
                    real_data.shape[0]
                )

                # -------------------------------------------------
                # Discriminator step
                # -------------------------------------------------

                discriminator_optimizer.zero_grad()

                real_logits = (
                    self.discriminator(
                        real_data
                    )
                )

                real_labels = torch.ones_like(
                    real_logits
                )

                real_loss = criterion(
                    real_logits,
                    real_labels
                )

                # Generate fake data
                with torch.no_grad():

                    fake_data = (
                        self.generate_from_condition(
                            condition
                        )
                    )

                fake_logits = (
                    self.discriminator(
                        fake_data
                    )
                )

                fake_labels = torch.zeros_like(
                    fake_logits
                )

                fake_loss = criterion(
                    fake_logits,
                    fake_labels
                )

                discriminator_loss = (
                    real_loss + fake_loss
                ) / 2

                discriminator_loss.backward()

                discriminator_optimizer.step()

                # -------------------------------------------------
                # Generator step
                # -------------------------------------------------

                generator_optimizer.zero_grad()

                fake_data = (
                    self.generate_from_condition(
                        condition
                    )
                )

                fake_logits = (
                    self.discriminator(
                        fake_data
                    )
                )

                generator_labels = (
                    torch.ones_like(
                        fake_logits
                    )
                )

                generator_loss = criterion(
                    fake_logits,
                    generator_labels
                )

                generator_loss.backward()

                generator_optimizer.step()

                generator_total += (
                    generator_loss.item()
                )

                discriminator_total += (
                    discriminator_loss.item()
                )

            average_generator = (
                generator_total / batches
            )

            average_discriminator = (
                discriminator_total / batches
            )

            history[
                "generator_loss"
            ].append(
                average_generator
            )

            history[
                "discriminator_loss"
            ].append(
                average_discriminator
            )

            if verbose:

                print(
                    f"Epoch "
                    f"{epoch + 1:03d}/"
                    f"{self.config.epochs} | "
                    f"G Loss: "
                    f"{average_generator:.4f} | "
                    f"D Loss: "
                    f"{average_discriminator:.4f}"
                )

        return history