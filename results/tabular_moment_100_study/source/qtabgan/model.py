"""Hybrid generator and optional label-conditioned discriminator (paper section 3.2).

Feature-only discrimination remains the default for historical checkpoint compatibility.
"""
import numpy as np
import pennylane as qml
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from .config import QTabGANConfig
from .quantum_generator import QuantumGeneratorCore
from .networks import ClassicalMapper, Discriminator
from .utils import set_seed


class QTabGAN(nn.Module):
    def __init__(self, data_dim, condition_dim, config=None):
        super().__init__()
        self.config = config or QTabGANConfig()
        config = self.config
        self.data_dim, self.condition_dim = data_dim, condition_dim
        if not np.isfinite(config.moment_loss_weight) or config.moment_loss_weight < 0:
            raise ValueError("moment_loss_weight must be finite and nonnegative")
        if not np.isfinite(config.categorical_temperature) or config.categorical_temperature <= 0:
            raise ValueError("categorical_temperature must be finite and positive")
        if config.tabular_output or config.moment_loss_weight:
            if (config.numerical_dim is None or config.numerical_dim < 0 or
                    any(width < 1 for width in config.categorical_dims) or
                    config.numerical_dim + sum(config.categorical_dims) != data_dim):
                raise ValueError("Output schema must cover data_dim in numeric/categorical order")
        set_seed(config.seed)
        self.quantum_generator = QuantumGeneratorCore(
            config.n_qubits, config.n_layers, config.shots, config.diff_method, config.seed,
            hadamards_per_layer=config.hadamards_per_layer,
            finite_shot_training=config.finite_shot_training, ansatz=config.ansatz)
        if config.classical_init_seed is not None:
            torch.manual_seed(config.classical_init_seed)
        self.mapper = ClassicalMapper(2 ** config.n_qubits, condition_dim,
                                      data_dim, config.mapper_hidden)
        self.discriminator = Discriminator(data_dim + (condition_dim if config.condition_discriminator else 0),
                                           config.discriminator_hidden)
        if not np.isfinite(config.quantum_init_gain) or config.quantum_init_gain <= 0:
            raise ValueError("quantum_init_gain must be finite and positive")
        self.quantum_initialization_mean = None
        if config.quantum_init_gain != 1.0:
            # Fold centering/scaling into initial W and b, leaving the paper's
            # raw [probabilities, label] -> ReLU MLP architecture unchanged.
            # Calibration is analytic and uses random latent inputs, not data rows.
            local_rng = torch.Generator().manual_seed(config.seed + 31000)
            z = torch.rand(256, config.n_qubits, generator=local_rng) * np.pi
            device = qml.device("default.qubit", wires=config.n_qubits, seed=config.seed + 31000)
            calibration = qml.QNode(self.quantum_generator._probabilities, device,
                                    interface="torch", diff_method=None)
            first = self.mapper.model.network[0]
            with torch.no_grad():
                mean = calibration(z, self.quantum_generator.weights).to(first.weight.dtype).mean(0)
                quantum_weights = first.weight[:, :2 ** config.n_qubits]
                old_mean_contribution = quantum_weights @ mean
                quantum_weights.mul_(config.quantum_init_gain)
                first.bias.add_((1.0 - config.quantum_init_gain) * old_mean_contribution)
                self.quantum_initialization_mean = mean.clone()
        self.to(config.device)

    def discriminate(self, features, condition):
        """D(x,y) when enabled; both real and generated rows use corresponding labels."""
        if self.config.condition_discriminator:
            features = torch.cat([features, condition], dim=1)
        return self.discriminator(features)

    def _noise(self, batch_size):
        return torch.rand(batch_size, self.config.n_qubits, device=self.config.device) * np.pi

    def generate_from_condition(self, condition, sample=False):
        noise = self._noise(len(condition))
        probabilities = (self.quantum_generator.sampled_probabilities(noise) if sample
                         else self.quantum_generator(noise))
        generated = self.mapper(probabilities, condition)
        if not self.config.tabular_output:
            return generated
        # Explicit extension to the paper's linear head. Straight-through samples
        # put real/fake categories on identical support while retaining gradients.
        numeric = self.config.numerical_dim
        parts = [np.pi * torch.sigmoid(generated[:, :numeric])]
        offset = numeric
        for width in self.config.categorical_dims:
            parts.append(F.gumbel_softmax(generated[:, offset:offset + width],
                         tau=self.config.categorical_temperature, hard=True, dim=1))
            offset += width
        return torch.cat(parts, dim=1)

    def generate(self, condition, sample=False):
        """Use the training probability QNode; sample=True uses shot histograms."""
        previous_mode = self.training
        self.eval()
        condition = torch.as_tensor(condition, dtype=torch.float32, device=self.config.device)
        try:
            with torch.no_grad():
                return self.generate_from_condition(condition, sample=sample).cpu().numpy()
        finally:
            self.train(previous_mode)

    def fit(self, data, conditions, verbose=True, epoch_callback=None):
        """Continue training, preserving Adam moments across fit calls."""
        self.train()
        data = torch.as_tensor(data, dtype=torch.float32, device=self.config.device)
        conditions = torch.as_tensor(conditions, dtype=torch.float32, device=self.config.device)
        if data.ndim != 2 or data.shape[1] != self.data_dim:
            raise ValueError("data must have shape (rows, data_dim)")
        if conditions.ndim != 2 or conditions.shape != (len(data), self.condition_dim) or not len(data):
            raise ValueError("conditions must have shape (nonzero rows, condition_dim)")
        if not torch.isfinite(data).all() or not torch.isfinite(conditions).all():
            raise ValueError("Training data and conditions must be finite")
        if self.config.epochs < 1 or self.config.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        loader = DataLoader(TensorDataset(data, conditions), batch_size=self.config.batch_size, shuffle=True)
        generator_parameters = list(self.quantum_generator.parameters()) + list(self.mapper.parameters())
        if not hasattr(self, '_generator_optimizer'):
            self._generator_optimizer = torch.optim.Adam(generator_parameters, lr=self.config.lr_generator)
            self._discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.lr_discriminator)
        g_opt, d_opt = self._generator_optimizer, self._discriminator_optimizer
        criterion = nn.BCEWithLogitsLoss()
        moment_loss = None
        if self.config.moment_loss_weight:
            from .regularization import ConditionalMomentLoss
            moment_loss = ConditionalMomentLoss(data, conditions, self.config.numerical_dim,
                                               self.config.categorical_dims)
        history = {key: [] for key in ['generator_loss', 'discriminator_loss', 'quantum_gradient_norm']}
        if moment_loss is not None:
            history['adversarial_loss'] = []
            history['moment_loss'] = []
        for epoch in range(self.config.epochs):
            totals = np.zeros(len(history))
            for real, condition in loader:
                d_opt.zero_grad(set_to_none=True)
                real_logits = self.discriminate(real, condition)
                with torch.no_grad():
                    fake = self.generate_from_condition(condition)
                fake_logits = self.discriminate(fake, condition)
                d_loss = (criterion(real_logits, torch.ones_like(real_logits)) +
                          criterion(fake_logits, torch.zeros_like(fake_logits))) / 2
                if not torch.isfinite(d_loss): raise FloatingPointError('Nonfinite discriminator loss')
                d_loss.backward(); d_opt.step()

                g_opt.zero_grad(set_to_none=True)
                generated = self.generate_from_condition(condition)
                logits = self.discriminate(generated, condition)
                adversarial_loss = criterion(logits, torch.ones_like(logits))
                g_loss = adversarial_loss
                if moment_loss is not None:
                    penalty = moment_loss(generated, condition)
                    g_loss = g_loss + self.config.moment_loss_weight * penalty
                if not torch.isfinite(g_loss): raise FloatingPointError('Nonfinite generator loss')
                g_loss.backward()
                gradient = self.quantum_generator.weights.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError('Missing or nonfinite quantum gradients')
                values = [g_loss.item(), d_loss.item(), gradient.norm().item()]
                if moment_loss is not None:
                    values += [adversarial_loss.item(), penalty.item()]
                totals += values
                g_opt.step()
            for key, value in zip(history, totals / len(loader)):
                history[key].append(float(value))
            if epoch_callback is not None:
                epoch_callback(epoch + 1, {key: values[-1] for key, values in history.items()})
            if verbose:
                print(f'Epoch {epoch+1:03d}/{self.config.epochs} | '
                      f'G Loss: {history["generator_loss"][-1]:.4f} | '
                      f'D Loss: {history["discriminator_loss"][-1]:.4f}', flush=True)
        return history
