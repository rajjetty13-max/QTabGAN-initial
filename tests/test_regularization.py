from dataclasses import asdict
import io
import unittest

import torch

from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.regularization import ConditionalMomentLoss


class ConditionalMomentTests(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([
            [0., 1., 1., 0.], [2., 3., 0., 1.],
            [4., 2., 1., 0.], [6., 4., 0., 1.],
        ], dtype=torch.float64)
        self.condition = torch.eye(2, dtype=torch.float64).repeat_interleave(2, dim=0)

    def test_training_moments_match_without_consuming_rng_or_attaching_targets(self):
        rng = torch.get_rng_state().clone()
        source = self.data.clone().requires_grad_()
        objective = ConditionalMomentLoss(source, self.condition, 2, [2])
        loss = objective(source, self.condition)
        self.assertLess(loss.item(), 1e-20)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(any(buffer.requires_grad for buffer in objective.buffers()))
        loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_swapping_classes_is_penalized_with_identical_global_moments(self):
        objective = ConditionalMomentLoss(self.data, self.condition, 2, [2])
        self.assertGreater(objective(self.data, self.condition.flip(1)).item(), 1.)

    def test_collapsed_numeric_has_positive_loss_and_finite_gradients(self):
        objective = ConditionalMomentLoss(self.data, self.condition, 2, [2])
        collapsed = self.data.clone()
        collapsed[:2, :2] = self.data[:2, :2].mean(0)
        collapsed[2:, :2] = self.data[2:, :2].mean(0)
        collapsed.requires_grad_()
        loss = objective(collapsed, self.condition)
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(collapsed.grad).all())

    def test_covariance_penalizes_wrong_relationship_with_matching_marginals(self):
        real = torch.tensor([[-1., -1.], [1., 1.]])
        fake = torch.tensor([[-1., 1.], [1., -1.]])
        objective = ConditionalMomentLoss(real, torch.ones(2, 1), 2, [])
        self.assertGreater(objective(fake, torch.ones(2, 1)).item(), 0.)

    def test_classes_and_categorical_blocks_have_equal_weights(self):
        # Adding unused ZIP categories does not dilute categorical discrepancies.
        real = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.]])
        fake = torch.tensor([[0., 1., 1., 0.], [0., 1., 1., 0.]])
        one_class = torch.ones(2, 1)
        base = ConditionalMomentLoss(real, one_class, 0, [2, 2])
        extended = ConditionalMomentLoss(torch.cat([real, torch.zeros(2, 100)], 1),
                                         one_class, 0, [2, 102])
        torch.testing.assert_close(base(fake, one_class),
                                   extended(torch.cat([fake, torch.zeros(2, 100)], 1), one_class))
        objective = ConditionalMomentLoss(self.data, self.condition, 2, [2])
        shifted = self.data.clone()
        shifted[:2, 0] += 1.
        full = objective(shifted, self.condition)
        # Repeating only the first class changes its frequency, not its weight.
        indices = torch.tensor([0, 1, 0, 1, 0, 1, 2, 3])
        torch.testing.assert_close(full, objective(shifted[indices], self.condition[indices]))

    def test_singleton_classes_skip_std_covariance_and_absent_batch_classes(self):
        objective = ConditionalMomentLoss(self.data, self.condition, 2, [2])
        # A singleton placed at its target mean has no variance estimate.
        singleton = self.data[:2].mean(0, keepdim=True).requires_grad_()
        loss = objective(singleton, self.condition[:1])
        self.assertLess(loss.item(), 1e-20)
        loss.backward()
        self.assertTrue(torch.isfinite(singleton.grad).all())

    def test_constant_training_features_and_quantum_generator_gradients_are_finite(self):
        config = QTabGANConfig(n_qubits=2, n_layers=1, ansatz="paper_readout",
                               mapper_hidden=(8,), discriminator_hidden=(8,))
        model = QTabGAN(3, 2, config)
        condition = torch.eye(2).repeat_interleave(3, dim=0)
        real = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.],
                             [0., 0., 1.], [0., 1., 0.], [0., 0., 1.]])
        objective = ConditionalMomentLoss(real, condition, 1, [2])
        fake = model.generate_from_condition(condition)
        loss = objective(fake, condition)
        loss.backward()
        gradient = model.quantum_generator.weights.grad
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm().item(), 0.)

    def test_fit_checkpoint_continuation_matches_uninterrupted_stochastic_training(self):
        options = dict(n_qubits=2, n_layers=1, ansatz="paper_readout",
                       mapper_hidden=(8,), discriminator_hidden=(8,), batch_size=4,
                       numerical_dim=2, categorical_dims=(2,), tabular_output=True,
                       condition_discriminator=True, moment_loss_weight=10.)
        real = self.data.float().clone()
        real[:, :2] /= 2
        condition = self.condition.float()
        full = QTabGAN(4, 2, QTabGANConfig(**options, epochs=2))
        full_history = full.fit(real, condition, verbose=False)
        self.assertIn("adversarial_loss", full_history)
        self.assertIn("moment_loss", full_history)
        self.assertTrue(all(value > 0 and torch.isfinite(torch.tensor(value))
                            for value in full_history["quantum_gradient_norm"]))
        for generator, adversarial, moment in zip(full_history["generator_loss"],
                                                  full_history["adversarial_loss"],
                                                  full_history["moment_loss"]):
            self.assertAlmostEqual(generator, adversarial + 10 * moment, places=5)

        first = QTabGAN(4, 2, QTabGANConfig(**options, epochs=1))
        first_history = first.fit(real, condition, verbose=False)
        serialized = io.BytesIO()
        torch.save({"config": asdict(first.config), "state_dict": first.state_dict(),
                    "generator_optimizer": first._generator_optimizer.state_dict(),
                    "discriminator_optimizer": first._discriminator_optimizer.state_dict(),
                    "rng": torch.get_rng_state()}, serialized)
        serialized.seek(0)
        saved = torch.load(serialized, weights_only=True)
        resumed = QTabGAN(4, 2, QTabGANConfig(**saved["config"]))
        resumed.load_state_dict(saved["state_dict"])
        generator_parameters = list(resumed.quantum_generator.parameters()) + list(resumed.mapper.parameters())
        resumed._generator_optimizer = torch.optim.Adam(generator_parameters, lr=resumed.config.lr_generator)
        resumed._discriminator_optimizer = torch.optim.Adam(resumed.discriminator.parameters(),
                                                            lr=resumed.config.lr_discriminator)
        resumed._generator_optimizer.load_state_dict(saved["generator_optimizer"])
        resumed._discriminator_optimizer.load_state_dict(saved["discriminator_optimizer"])
        torch.set_rng_state(saved["rng"])
        resumed_history = resumed.fit(real, condition, verbose=False)
        for key in full_history:
            self.assertEqual(full_history[key], first_history[key] + resumed_history[key])
        for key, weight in full.state_dict().items():
            torch.testing.assert_close(weight, resumed.state_dict()[key], rtol=0, atol=0)

    def test_invalid_schema_and_conditions_are_rejected(self):
        for numerical, categorical in [(-1, [2]), (2, [0]), (2, [3]), (True, [3])]:
            with self.subTest(numerical=numerical, categorical=categorical), self.assertRaises(ValueError):
                ConditionalMomentLoss(self.data, self.condition, numerical, categorical)
        bad = self.condition.clone()
        bad[0] = torch.tensor([.5, .5])
        with self.assertRaises(ValueError):
            ConditionalMomentLoss(self.data, bad, 2, [2])
        with self.assertRaises(ValueError):
            ConditionalMomentLoss(self.data[:2], self.condition[:2], 2, [2])
        objective = ConditionalMomentLoss(self.data, self.condition, 2, [2])
        for fake, condition in [(self.data[:0], self.condition[:0]),
                                (self.data, bad), (self.data[:, :3], self.condition),
                                (self.data * float("nan"), self.condition)]:
            with self.assertRaises(ValueError):
                objective(fake, condition)


if __name__ == "__main__":
    unittest.main()
