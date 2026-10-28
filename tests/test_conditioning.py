import unittest
import numpy as np
import torch
from qtabgan import QTabGAN, QTabGANConfig


class ConditioningTests(unittest.TestCase):
    def test_matching_labels_reach_all_three_adversarial_passes(self):
        cfg = QTabGANConfig(n_qubits=2, n_layers=1, epochs=1, batch_size=4,
                           mapper_hidden=(8,), discriminator_hidden=(8,), condition_discriminator=True)
        model = QTabGAN(2,2,cfg)
        x = np.array([[0,.1],[1,.2],[2,.3],[3,.4]],dtype=np.float32)
        y = np.eye(2,dtype=np.float32)[[0,1,0,1]]
        seen=[]
        hook=model.discriminator.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
        history=model.fit(x,y,verbose=False)
        hook.remove()
        self.assertEqual(len(seen),3)
        expected=torch.tensor(y)[seen[0][:,0].long()]
        for values in seen:
            self.assertEqual(values.shape,(4,4))
            torch.testing.assert_close(values[:,-2:],expected)
        self.assertGreater(history['quantum_gradient_norm'][0],0)
        fixed=torch.zeros(2,2)
        self.assertFalse(torch.allclose(model.discriminate(fixed,torch.eye(2))[0],
                                        model.discriminate(fixed,torch.eye(2))[1]))

    def test_generator_initialization_unchanged_and_legacy_shapes_preserved(self):
        base=dict(n_qubits=2,mapper_hidden=(8,),discriminator_hidden=(8,))
        old=QTabGAN(3,2,QTabGANConfig(**base))
        new=QTabGAN(3,2,QTabGANConfig(**base,condition_discriminator=True))
        for key,weight in old.mapper.state_dict().items():
            torch.testing.assert_close(weight,new.mapper.state_dict()[key],rtol=0,atol=0)
        torch.testing.assert_close(old.quantum_generator.weights,new.quantum_generator.weights,rtol=0,atol=0)
        self.assertEqual(old.discriminator.model.network[0].in_features,3)
        self.assertEqual(new.discriminator.model.network[0].in_features,5)


if __name__=='__main__': unittest.main()
