import unittest
import numpy as np
import torch
import pennylane as qml
from qtabgan.quantum_generator import QuantumGeneratorCore
from qtabgan.ionq import export_qis,simulate_qis


class ReadoutTests(unittest.TestCase):
    def test_only_added_operations_are_final_hadamards_and_phases_become_visible(self):
        torch.manual_seed(73)
        base=QuantumGeneratorCore(4,2).double()
        new=QuantumGeneratorCore(4,2,ansatz='paper_readout').double()
        new.load_state_dict(base.state_dict())
        z=torch.rand(3,4,dtype=torch.float64)*np.pi
        old_tape=qml.tape.make_qscript(base._probabilities)(z[0],base.weights)
        new_tape=qml.tape.make_qscript(new._probabilities)(z[0],new.weights)
        self.assertEqual(len(new_tape.operations),len(old_tape.operations)+4)
        for a,b in zip(old_tape.operations,new_tape.operations):
            self.assertEqual(a.name,b.name);self.assertEqual(a.wires,b.wires)
            for pa,pb in zip(a.parameters,b.parameters):torch.testing.assert_close(pa,pb)
        self.assertTrue(all(op.name=='Hadamard' for op in new_tape.operations[-4:]))
        cost=torch.randn(3,16,dtype=torch.float64)
        (new.circuit(z,new.weights)*cost).sum().backward()
        self.assertTrue((new.weights.grad.abs()>1e-8).all())
        q=export_qis(new,z[0])
        np.testing.assert_allclose(simulate_qis(q),new.circuit(z[0],new.weights).detach().numpy(),atol=1e-12)

    def test_readout_backprop_matches_parameter_shift_and_shot_path(self):
        torch.manual_seed(12)
        back=QuantumGeneratorCore(2,2,ansatz='paper_readout',hadamards_per_layer=True,shots=256)
        shift=QuantumGeneratorCore(2,2,ansatz='paper_readout',hadamards_per_layer=True,diff_method='parameter-shift')
        shift.load_state_dict(back.state_dict())
        z=torch.rand(2,2)*np.pi;cost=torch.tensor([0.,1.,3.,7.])
        a,b=back(z),shift(z)
        torch.testing.assert_close(a,b)
        (a*cost).sum().backward();(b*cost).sum().backward()
        torch.testing.assert_close(back.weights.grad,shift.weights.grad,atol=2e-6,rtol=1e-4)
        sampled=back.sampled_probabilities(z)
        torch.testing.assert_close(sampled.sum(1),torch.ones(2))
        torch.testing.assert_close(sampled*256,(sampled*256).round())


if __name__=='__main__':unittest.main()
