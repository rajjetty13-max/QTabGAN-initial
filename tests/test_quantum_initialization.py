import unittest
import numpy as np
import torch
from qtabgan import QTabGAN,QTabGANConfig


class QuantumInitializationTests(unittest.TestCase):
    def test_center_is_preserved_variations_scaled_other_weights_and_rng_unchanged(self):
        options=dict(n_qubits=4,n_layers=2,mapper_hidden=(12,8),discriminator_hidden=(8,),
                     condition_discriminator=True,classical_init_seed=42042)
        base=QTabGAN(3,2,QTabGANConfig(**options))
        torch_state=torch.get_rng_state().clone();np_next=np.random.random()
        gained=QTabGAN(3,2,QTabGANConfig(**options,quantum_init_gain=16))
        torch.testing.assert_close(torch.get_rng_state(),torch_state,rtol=0,atol=0)
        self.assertEqual(np.random.random(),np_next)
        torch.testing.assert_close(base.quantum_generator.weights,gained.quantum_generator.weights,rtol=0,atol=0)
        for key,value in base.discriminator.state_dict().items():
            torch.testing.assert_close(value,gained.discriminator.state_dict()[key],rtol=0,atol=0)
        mean=gained.quantum_initialization_mean
        c=torch.tensor([[1.,0.],[0.,1.]])
        center=torch.cat([mean.expand(2,-1),c],dim=1)
        first0=base.mapper.model.network[0];first1=gained.mapper.model.network[0]
        torch.testing.assert_close(first0(center),first1(center),atol=2e-7,rtol=1e-5)
        delta=torch.linspace(-.001,.001,16).expand(2,-1)
        changed=center.clone();changed[:,:16]+=delta
        # Compare perturbations in double precision to avoid subtracting
        # nearly equal float32 preactivations and amplifying rounding by 16.
        linear=torch.nn.functional.linear
        diff1=linear(changed.double(),first1.weight.double(),first1.bias.double())-linear(center.double(),first1.weight.double(),first1.bias.double())
        diff0=linear(changed.double(),first0.weight.double(),first0.bias.double())-linear(center.double(),first0.weight.double(),first0.bias.double())
        torch.testing.assert_close(diff1,16*diff0,atol=1e-12,rtol=1e-10)
        for key,value in base.mapper.state_dict().items():
            if key not in ['model.network.0.weight','model.network.0.bias']:
                torch.testing.assert_close(value,gained.mapper.state_dict()[key],rtol=0,atol=0)
        # The folded initialization needs no additional buffers to load/infer.
        clone=QTabGAN(3,2,QTabGANConfig(**options,quantum_init_gain=16))
        clone.load_state_dict(gained.state_dict())
        z=torch.rand(2,4)
        out=gained.mapper(gained.quantum_generator(z),c)
        torch.testing.assert_close(out,clone.mapper(clone.quantum_generator(z),c))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(gained.quantum_generator.weights.grad).all())
        self.assertGreater(gained.quantum_generator.weights.grad.norm().item(),0)

    def test_conditional_gained_checkpoint_loads_for_hardware_inference(self):
        from dataclasses import asdict
        from pathlib import Path
        import tempfile
        from scripts.ionq_generate import load_checkpoint
        cfg=QTabGANConfig(n_qubits=2,n_layers=1,ansatz='paper_readout',condition_discriminator=True,quantum_init_gain=16)
        model=QTabGAN(3,2,cfg)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt'
            torch.save({'state_dict':model.state_dict(),'config':asdict(cfg)},path)
            loaded,_=load_checkpoint(path)
        self.assertEqual(loaded.data_dim,3)
        self.assertEqual(loaded.discriminator.model.network[0].in_features,5)
        z=torch.rand(2,2);c=torch.eye(2)
        with torch.no_grad():
            torch.testing.assert_close(model.mapper(model.quantum_generator(z),c),loaded.mapper(loaded.quantum_generator(z),c))

    def test_invalid_gain_is_rejected(self):
        for gain in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):QTabGAN(3,2,QTabGANConfig(n_qubits=2,quantum_init_gain=gain))


if __name__=='__main__':unittest.main()
