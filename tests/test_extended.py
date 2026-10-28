"""Restart correctness and diagnostic isolation; CPU analytic training."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from qtabgan import QTabGAN, QTabGANConfig
from extended_benchmark import rng_state, restore_rng, evaluation_rng, atomic_save


class RestartTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self):
        torch.set_num_threads(1)
        x = np.arange(24, dtype=np.float32).reshape(8, 3)/24
        conditions = np.eye(2, dtype=np.float32)[np.arange(8)%2]
        cfg = dict(n_qubits=2, n_layers=1, batch_size=4, seed=17,
                   mapper_hidden=(4,), discriminator_hidden=(4,))
        full = QTabGAN(3, 2, QTabGANConfig(epochs=2, **cfg))
        full.fit(x, conditions, verbose=False)
        split = QTabGAN(3, 2, QTabGANConfig(epochs=1, **cfg))
        observed = []
        split.fit(x, conditions, verbose=False, epoch_callback=lambda e, row: observed.append((e,row)))
        checkpoint = {'model': split.state_dict(), 'g': split._generator_optimizer.state_dict(),
                      'd': split._discriminator_optimizer.state_dict(), 'rng': rng_state()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            atomic_save(checkpoint, path)
            checkpoint = torch.load(path, weights_only=True)
        resumed = QTabGAN(3, 2, QTabGANConfig(epochs=1, **cfg))
        resumed.load_state_dict(checkpoint['model'])
        gp = list(resumed.quantum_generator.parameters()) + list(resumed.mapper.parameters())
        resumed._generator_optimizer = torch.optim.Adam(gp)
        resumed._discriminator_optimizer = torch.optim.Adam(resumed.discriminator.parameters())
        resumed._generator_optimizer.load_state_dict(checkpoint['g'])
        resumed._discriminator_optimizer.load_state_dict(checkpoint['d'])
        restore_rng(checkpoint['rng'])
        with evaluation_rng(123):
            resumed.generate(conditions)
            np.random.normal(size=100)
        resumed.fit(x, conditions, verbose=False)
        for name, weight in full.state_dict().items():
            torch.testing.assert_close(weight, resumed.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(observed[0][0], 1)
        self.assertGreater(observed[0][1]['quantum_gradient_norm'], 0)


if __name__ == '__main__': unittest.main()
