"""Optional class-conditional moment matching against training data only."""

from numbers import Integral

import torch
from torch import nn


class ConditionalMomentLoss(nn.Module):
    """Match numerical moments and categorical marginals within each class.

    ``data`` contains numerical columns followed by one categorical block per
    entry of ``categorical_dims``. Targets must be fitted on training rows only.
    Numerical errors use the training-wide population standard deviation, with
    a floor of 1e-4. Each categorical block contributes one squared vector
    distance, irrespective of its number of categories. Classes represented in
    a generated batch receive equal weight rather than frequency weighting.

    This is an optional training objective, not part of the paper's GAN loss.
    All targets are detached buffers; constructing or evaluating the loss does
    not consume random numbers.
    """

    def __init__(self, data, conditions, numerical_dim, categorical_dims):
        super().__init__()
        if isinstance(numerical_dim, bool) or not isinstance(numerical_dim, Integral) or numerical_dim < 0:
            raise ValueError("numerical_dim must be a nonnegative integer")
        categorical_dims = tuple(categorical_dims)
        if any(isinstance(width, bool) or not isinstance(width, Integral) or width < 1
               for width in categorical_dims):
            raise ValueError("categorical_dims must contain positive integer widths")
        self.numerical_dim = int(numerical_dim)
        self.categorical_dims = tuple(int(width) for width in categorical_dims)
        self.data_dim = self.numerical_dim + sum(self.categorical_dims)
        if not self.data_dim:
            raise ValueError("At least one feature is required")

        data = torch.as_tensor(data).detach()
        if not data.is_floating_point():
            data = data.float()
        if data.ndim != 2 or not len(data) or data.shape[1] != self.data_dim:
            raise ValueError("Training data must have shape (nonzero rows, schema width)")
        if not torch.isfinite(data).all():
            raise ValueError("Training data must be finite")
        conditions = torch.as_tensor(conditions, device=data.device)
        self._validate_conditions(conditions, len(data))
        self.condition_dim = conditions.shape[1]
        classes = conditions.argmax(dim=1)
        counts = torch.bincount(classes, minlength=self.condition_dim)
        if (counts == 0).any():
            raise ValueError("Every conditioning class must have training rows")

        numeric = data[:, :self.numerical_dim]
        center = numeric.mean(dim=0)
        scale = (numeric.std(dim=0, correction=0).clamp_min(1e-4)
                 if self.numerical_dim else numeric.new_empty((0,)))
        standardized = (numeric - center) / scale
        means, stds, covariances, categorical_means = [], [], [], []
        for label in range(self.condition_dim):
            rows = classes == label
            values = standardized[rows]
            mean = values.mean(dim=0)
            centered = values - mean
            means.append(mean)
            stds.append(values.std(dim=0, correction=0)
                        if self.numerical_dim else values.new_empty((0,)))
            covariances.append(centered.T @ centered / len(values))
            categorical_means.append(data[rows, self.numerical_dim:].mean(dim=0))
        self.register_buffer("numeric_center", center.clone())
        self.register_buffer("numeric_scale", scale.clone())
        self.register_buffer("target_numeric_mean", torch.stack(means))
        self.register_buffer("target_numeric_std", torch.stack(stds))
        self.register_buffer("target_numeric_covariance", torch.stack(covariances))
        self.register_buffer("target_categorical_mean", torch.stack(categorical_means))
        self.register_buffer("class_counts", counts)

    @staticmethod
    def _validate_conditions(conditions, rows, width=None):
        if conditions.ndim != 2 or conditions.shape[0] != rows or conditions.shape[1] < 1:
            raise ValueError("Conditions must have shape (rows, nonzero classes)")
        if width is not None and conditions.shape[1] != width:
            raise ValueError("Condition width differs from training targets")
        if not torch.isfinite(conditions).all():
            raise ValueError("Conditions must be finite one-hot rows")
        if not (((conditions == 0) | (conditions == 1)).all() and
                (conditions.sum(dim=1) == 1).all()):
            raise ValueError("Conditions must be one-hot rows")

    def forward(self, fake, condition):
        if not isinstance(fake, torch.Tensor) or not fake.is_floating_point():
            raise ValueError("Generated features must be a floating-point tensor")
        if fake.ndim != 2 or not len(fake) or fake.shape[1] != self.data_dim:
            raise ValueError("Generated features must have shape (nonzero rows, schema width)")
        if not torch.isfinite(fake).all():
            raise ValueError("Generated features must be finite")
        condition = torch.as_tensor(condition, device=fake.device)
        self._validate_conditions(condition, len(fake), self.condition_dim)
        if fake.device != self.numeric_scale.device:
            raise ValueError("Move the moment loss and generated features to the same device")

        classes = condition.argmax(dim=1)
        standardized = (fake[:, :self.numerical_dim] - self.numeric_center) / self.numeric_scale
        class_losses = []
        for label in classes.unique():
            rows = classes == label
            # Keep even a feature-empty family's zero attached to the graph.
            loss = fake[rows].sum() * 0.0
            if self.numerical_dim:
                values = standardized[rows]
                mean = values.mean(dim=0)
                loss = loss + (mean - self.target_numeric_mean[label]).square().mean()
                if len(values) >= 2:
                    # torch.std defines a finite, zero gradient at exact collapse.
                    std = values.std(dim=0, correction=0)
                    centered = values - mean
                    covariance = centered.T @ centered / len(values)
                    loss = loss + (std - self.target_numeric_std[label]).square().mean()
                    loss = loss + (covariance - self.target_numeric_covariance[label]).square().mean()
            if self.categorical_dims:
                marginal = fake[rows, self.numerical_dim:].mean(dim=0)
                difference = marginal - self.target_categorical_mean[label]
                block_losses = [block.square().sum()
                                for block in difference.split(self.categorical_dims)]
                loss = loss + torch.stack(block_losses).mean()
            class_losses.append(loss)
        return torch.stack(class_losses).mean()
