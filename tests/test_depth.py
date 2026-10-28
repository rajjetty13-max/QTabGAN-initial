import unittest
import numpy as np
import torch
import pennylane as qml
from qtabgan import QTabGAN,QTabGANConfig
from qtabgan.ionq import export_qis,simulate_qis


class DepthTests(unittest.TestCase):
    def test_classical_initialization_is_fixed_when_quantum_depth_changes(self):
        models=[QTabGAN(3,2,QTabGANConfig(n_qubits=4,n_layers=depth,ansatz='paper_readout',
                                        condition_discriminator=True,classical_init_seed=42042)) for depth in [2,3]]
        for module in ['mapper','discriminator']:
            for key,value in getattr(models[0],module).state_dict().items():
                torch.testing.assert_close(value,getattr(models[1],module).state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(models[0].quantum_generator.weights,models[1].quantum_generator.weights[:2],rtol=0,atol=0)
        core=models[1].quantum_generator.double()
        z=torch.rand(2,4,dtype=torch.float64)*np.pi
        p=core.circuit(z,core.weights)
        tape=qml.tape.make_qscript(core._probabilities)(z[0],core.weights)
        self.assertEqual(sum(op.name=='CNOT' for op in tape.operations),12)
        self.assertEqual(core.weights.numel(),24)
        cost=torch.randn(2,16,dtype=torch.float64)
        (p*cost).sum().backward()
        self.assertTrue((core.weights.grad.abs()>1e-9).all())
        np.testing.assert_allclose(simulate_qis(export_qis(core,z[0])),p[0].detach().numpy(),atol=1e-12)

    def test_one_layer_readout_keeps_all_angles_effective(self):
        torch.manual_seed(19)
        model=QTabGAN(3,2,QTabGANConfig(n_qubits=4,n_layers=1,ansatz='paper_readout'))
        core=model.quantum_generator.double()
        z=torch.rand(3,4,dtype=torch.float64)*np.pi
        probabilities=core.circuit(z,core.weights)
        (probabilities*torch.randn(3,16,dtype=torch.float64)).sum().backward()
        self.assertTrue((core.weights.grad.abs()>1e-9).all())
        np.testing.assert_allclose(simulate_qis(export_qis(core,z[0])),probabilities[0].detach().numpy(),atol=1e-12)


if __name__=='__main__':unittest.main()
