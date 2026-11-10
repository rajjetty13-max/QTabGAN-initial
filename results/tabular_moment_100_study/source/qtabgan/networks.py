import torch
from torch import nn


class MLP(nn.Module):
    """
    Generic feed-forward neural network.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        output_activation=None
    ):
        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                output_dim
            )
        )

        if output_activation is not None:

            layers.append(
                output_activation
            )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(x)


class ClassicalMapper(nn.Module):
    """
    Maps the quantum probability vector and condition
    into synthetic tabular data.
    """

    def __init__(
        self,
        quantum_dim,
        condition_dim,
        output_dim,
        hidden_dims=(128, 128)
    ):
        super().__init__()

        self.model = MLP(
            input_dim=(
                quantum_dim
                + condition_dim
            ),
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )

    def forward(
        self,
        quantum_probability,
        condition
    ):

        x = torch.cat(
            [
                quantum_probability,
                condition
            ],
            dim=1
        )

        return self.model(x)


class Discriminator(nn.Module):
    """
    Classical discriminator.

    Outputs a single logit:
        positive -> real
        negative -> fake
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 64, 32)
    ):
        super().__init__()

        self.model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1
        )

    def forward(self, x):

        return self.model(x)