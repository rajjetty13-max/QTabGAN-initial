import unittest
import numpy as np
import pandas as pd
import torch
import pennylane as qml
from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.quantum_generator import QuantumGeneratorCore
from qtabgan.preprocessing import TabularPreprocessor
from qtabgan.evaluation import classification_utility, correlation_difference


class CoreTests(unittest.TestCase):
    def test_quantum_batch_gradient_and_shots(self):
        torch.manual_seed(7)
        core = QuantumGeneratorCore(3, 2, shots=128)
        noise = torch.rand(4, 3)
        batch = core(noise)
        scalar = torch.stack([core.circuit(row, core.weights) for row in noise]).float()
        torch.testing.assert_close(batch, scalar)
        torch.testing.assert_close(batch.sum(1), torch.ones(4))
        (batch * torch.arange(8)).sum().backward()
        self.assertTrue(torch.isfinite(core.weights.grad).all())
        self.assertGreater(core.weights.grad.norm().item(), 1e-6)
        sampled = core.sampled_probabilities(noise)
        torch.testing.assert_close(sampled.sum(1), torch.ones(4))
        torch.testing.assert_close(sampled * 128, (sampled * 128).round())

    def test_finite_shot_training_probabilities_and_gradients(self):
        core = QuantumGeneratorCore(2, 1, shots=128, diff_method='parameter-shift',
                                    finite_shot_training=True)
        probabilities = core(torch.rand(2, 2))
        torch.testing.assert_close(probabilities * 128, (probabilities * 128).round())
        (probabilities * torch.arange(4)).sum().backward()
        self.assertTrue(torch.isfinite(core.weights.grad).all())
        self.assertGreater(core.weights.grad.norm().item(), 0.)
        with self.assertRaises(ValueError):
            QuantumGeneratorCore(finite_shot_training=True, diff_method='backprop')

    def test_repeated_hadamards_are_an_explicit_architecture_option(self):
        once = QuantumGeneratorCore(3, 2)
        repeated = QuantumGeneratorCore(3, 2, hadamards_per_layer=True)
        repeated.load_state_dict(once.state_dict())
        noise = torch.rand(2, 3)
        for core, expected in [(once, 3), (repeated, 6)]:
            tape = qml.tape.make_qscript(core._probabilities)(noise[0], core.weights)
            self.assertEqual(sum(g.name == 'Hadamard' for g in tape.operations), expected)
        self.assertFalse(torch.allclose(once(noise), repeated(noise)))
        repeated(noise).sum().backward()
        self.assertTrue(torch.isfinite(repeated.weights.grad).all())

    def test_shot_bit_order_matches_probability_columns(self):
        core = QuantumGeneratorCore(3, 1, shots=64)
        with torch.no_grad(): core.weights.zero_()
        # H followed by RY(+pi/2) prepares 1; -pi/2 prepares 0.
        # The ordered ring maps 100 -> 110 -> 111 -> 011 (index 3).
        noise = torch.tensor([[np.pi/2, -np.pi/2, -np.pi/2]])
        expected = torch.zeros(1, 8); expected[0, 3] = 1.
        torch.testing.assert_close(core(noise), expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(core.sampled_probabilities(noise), expected)

    def test_parameter_shift_matches_backprop(self):
        core = QuantumGeneratorCore(2, 1)
        shift = QuantumGeneratorCore(2, 1, diff_method='parameter-shift')
        shift.load_state_dict(core.state_dict())
        noise = torch.rand(2, 2)
        cost = torch.tensor([0., 1., 2., 4.])
        (core(noise) * cost).sum().backward()
        (shift(noise) * cost).sum().backward()
        torch.testing.assert_close(core.weights.grad, shift.weights.grad, atol=1e-5, rtol=1e-4)

    def test_correlation_uses_unique_pairs_without_diagonal(self):
        real = np.array([[0.,0.],[1.,1.],[2.,2.]])
        fake = np.array([[0.,2.],[1.,1.],[2.,0.]])
        self.assertAlmostEqual(correlation_difference(real, fake), 2.)
        self.assertIsNone(correlation_difference(np.ones((3,2)), np.ones((3,2))))

    def test_training_labels_are_independent_of_test_labels(self):
        x = np.array([[0], [0], [1], [1]], dtype=float)
        y = np.array([0,0,1,1])
        scores = classification_utility(x, x, x, 1-y, real_labels=y, synthetic_labels=y)
        self.assertEqual(scores['real_train_accuracy'], 0.)
        self.assertEqual(scores['synthetic_train_accuracy'], 0.)

    def test_fit_updates_quantum_parameters_and_keeps_optimizer(self):
        model = QTabGAN(2, 2, QTabGANConfig(n_qubits=2, n_layers=1,
                         epochs=1, batch_size=4, mapper_hidden=(8,),
                         discriminator_hidden=(8,)))
        data = np.array([[.1,.4],[.2,.3],[.8,.6],[.9,.7]], dtype=np.float32)
        condition = np.eye(2, dtype=np.float32)[[0,0,1,1]]
        before = model.quantum_generator.weights.detach().clone()
        history = model.fit(data, condition, verbose=False)
        self.assertGreater(history['quantum_gradient_norm'][0], 0.)
        self.assertGreater((before-model.quantum_generator.weights).abs().max().item(), 0.)
        optimizer = model._generator_optimizer
        model.generate(condition, sample=True)
        self.assertTrue(model.training)
        model.fit(data, condition, verbose=False)
        self.assertIs(model._generator_optimizer, optimizer)

    def test_preprocessor_refit_and_holdout(self):
        pre = TabularPreprocessor(['x'], ['category'], noise_std=0)
        frame = pd.DataFrame({'x':[0.,1.,2.], 'category':['a','b','a']})
        pre.fit(frame); first = list(pre.output_columns); pre.fit(frame)
        self.assertEqual(first, pre.output_columns)
        params = dict(pre.mean)
        pre.transform(pd.DataFrame({'x':[100.], 'category':['new']}))
        self.assertEqual(params, pre.mean)
        result = pre.inverse_transform(pre.transform(frame))
        self.assertEqual(result.category.tolist(), frame.category.tolist())


if __name__ == '__main__': unittest.main()
