"""Behavioral checks for the optional mixed-type generator output."""
from dataclasses import asdict
import io
import unittest

import numpy as np
import torch

from qtabgan import QTabGAN, QTabGANConfig


class TabularOutputTests(unittest.TestCase):
    def config(self, **overrides):
        options = dict(n_qubits=2, n_layers=1, ansatz='paper_readout',
                       mapper_hidden=(12,), discriminator_hidden=(8,),
                       tabular_output=True, numerical_dim=2,
                       categorical_dims=(2, 3))
        options.update(overrides)
        return QTabGANConfig(**options)

    def assert_valid_table(self, output):
        output = torch.as_tensor(output)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.shape[1], 7)
        self.assertTrue((output[:, :2] >= 0).all())
        self.assertTrue((output[:, :2] <= np.pi).all())
        for block in (output[:, 2:4], output[:, 4:7]):
            self.assertTrue(((block == 0) | (block == 1)).all())
            torch.testing.assert_close(block.sum(dim=1), torch.ones(len(block)),
                                       rtol=0, atol=0)

    def test_training_and_inference_outputs_have_valid_feature_support(self):
        model = QTabGAN(7, 2, self.config())
        conditions = torch.eye(2).repeat(12, 1)
        for training in (True, False):
            with self.subTest(training=training):
                model.train(training)
                self.assert_valid_table(model.generate_from_condition(conditions))
                self.assert_valid_table(model.generate(conditions))
                self.assertEqual(model.training, training)
        # Shot-based inference must pass through the same output contract.
        self.assert_valid_table(model.generate(conditions[:4], sample=True))

    def test_categorical_samples_retain_mapper_and_quantum_gradients(self):
        model = QTabGAN(7, 2, self.config())
        conditions = torch.eye(2).repeat(8, 1)
        output = model.generate_from_condition(conditions)
        # A category sum is constant and cannot demonstrate a useful gradient.
        # Distinct costs test learning through the hard category assignments.
        costs = torch.tensor([0., 0., -1., 2., -2., .5, 3.])
        (output * costs).sum().backward()
        quantum_gradient = model.quantum_generator.weights.grad
        self.assertIsNotNone(quantum_gradient)
        self.assertTrue(torch.isfinite(quantum_gradient).all())
        self.assertGreater(quantum_gradient.norm().item(), 0)
        mapper_gradients = [parameter.grad for parameter in model.mapper.parameters()]
        self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                            for gradient in mapper_gradients))
        self.assertGreater(sum(gradient.norm().item() for gradient in mapper_gradients), 0)

    def test_invalid_output_schema_temperature_and_moment_weight_are_rejected(self):
        invalid = [
            dict(numerical_dim=None),
            dict(numerical_dim=-1),
            dict(numerical_dim=True, categorical_dims=(2, 4)),
            dict(numerical_dim=2.0),
            dict(numerical_dim=8),
            dict(categorical_dims=(2, 2)),
            dict(categorical_dims=(0, 5)),
            dict(categorical_dims=(-1, 6)),
            dict(categorical_dims=(2.0, 3.0)),
            dict(categorical_dims=(True, 4)),
        ]
        invalid.extend(dict(categorical_temperature=value)
                       for value in (0, -1, float('nan'), float('inf')))
        invalid.extend(dict(moment_loss_weight=value)
                       for value in (-1, float('nan'), float('inf')))
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    QTabGAN(7, 2, self.config(**overrides))

    def test_legacy_configuration_and_weights_keep_linear_output(self):
        options = dict(n_qubits=2, n_layers=1, ansatz='paper_readout',
                       mapper_hidden=(12,), discriminator_hidden=(8,))
        original = QTabGAN(7, 2, QTabGANConfig(**options))
        legacy_config = asdict(original.config)
        for key in ('tabular_output', 'numerical_dim', 'categorical_dims',
                    'categorical_temperature', 'moment_loss_weight'):
            legacy_config.pop(key)
        checkpoint = io.BytesIO()
        torch.save({'config': legacy_config, 'state_dict': original.state_dict()}, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=True)
        clone = QTabGAN(7, 2, QTabGANConfig(**restored['config']))
        clone.load_state_dict(restored['state_dict'], strict=True)
        conditions = torch.eye(2).repeat(2, 1)
        torch.manual_seed(902)
        with torch.no_grad():
            expected = original.mapper(original.quantum_generator(original._noise(4)), conditions)
        torch.manual_seed(902)
        actual = clone.generate(conditions)
        np.testing.assert_array_equal(actual, expected.numpy())
        self.assertFalse(clone.config.tabular_output)


if __name__ == '__main__':
    unittest.main()
