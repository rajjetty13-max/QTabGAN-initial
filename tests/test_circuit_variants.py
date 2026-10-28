import unittest
import numpy as np
import pennylane as qml
import torch
from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.quantum_generator import QuantumGeneratorCore
from qtabgan.ionq import export_qis, simulate_qis


class CircuitVariantTests(unittest.TestCase):
    def test_final_phase_is_invisible_in_original_and_visible_in_candidate(self):
        torch.manual_seed(42)
        noise = torch.rand(3,4,dtype=torch.float64)*np.pi
        cost = torch.randn(3,16,dtype=torch.float64)
        for ansatz in ['paper','reuploading']:
            core = QuantumGeneratorCore(4,2,ansatz=ansatz).double()
            (core.circuit(noise,core.weights)*cost).sum().backward()
            if ansatz == 'paper':
                self.assertLess(core.weights.grad[-1,:,1].abs().max().item(),1e-12)
            else:
                self.assertTrue((core.weights.grad.abs()>1e-8).all())

    def test_candidate_batch_gradient_and_export_agree(self):
        torch.manual_seed(21)
        core = QuantumGeneratorCore(4,2,ansatz='reuploading')
        shift = QuantumGeneratorCore(4,2,ansatz='reuploading',diff_method='parameter-shift')
        shift.load_state_dict(core.state_dict())
        noise = torch.rand(2,4)*np.pi
        batched = core(noise)
        scalar = torch.stack([core.circuit(row,core.weights) for row in noise]).float()
        torch.testing.assert_close(batched,scalar)
        torch.testing.assert_close(batched.sum(1),torch.ones(2))
        cost = torch.arange(16,dtype=torch.float32)**2
        (batched*cost).sum().backward();(shift(noise)*cost).sum().backward()
        torch.testing.assert_close(core.weights.grad,shift.weights.grad,atol=3e-5,rtol=1e-4)
        exported = simulate_qis(export_qis(core,noise[0]))
        np.testing.assert_allclose(exported,batched[0].detach().numpy(),atol=2e-7)

    def test_config_selects_variant_and_old_configs_default_to_paper(self):
        model = QTabGAN(3,2,QTabGANConfig(ansatz='reuploading',n_qubits=4))
        self.assertEqual(model.quantum_generator.ansatz,'reuploading')
        self.assertEqual(QTabGANConfig().ansatz,'paper')
        with self.assertRaises(ValueError): QuantumGeneratorCore(3,2,ansatz='reuploading')
        with self.assertRaises(ValueError): QuantumGeneratorCore(4,2,ansatz='reuploading',hadamards_per_layer=True)


if __name__=='__main__': unittest.main()
