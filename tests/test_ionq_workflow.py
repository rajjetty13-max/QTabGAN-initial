"""Complete inference workflow using an in-memory IonQ service fixture."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import joblib
import numpy as np
import pandas as pd
import torch

from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.preprocessing import TabularPreprocessor
from qtabgan.ionq import collect_bundle, load_bundle, simulate_qis, submit_bundle, write_json
from scripts.ionq_generate import local, map_probabilities, prepare


class MemoryIonQ:
    """Return flat decimal-key results using IonQ's documented little-endian convention."""
    def __init__(self):
        self.jobs = {}
        self.submissions = 0

    def submit(self, payload):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = payload
        self.submissions += 1
        return {'id': job_id, 'status': 'submitted'}

    def job(self, job_id):
        payload = self.jobs[job_id]
        return {'id': job_id, 'name': payload['name'], 'backend': payload['backend'], 'status': 'completed'}

    def probabilities(self, job_id, job):
        circuit = self.jobs[job_id]['input']
        n = circuit['qubits']
        values = simulate_qis(circuit)
        return {str(int(format(i,f'0{n}b')[::-1],2)):float(p) for i,p in enumerate(values) if p > 0}


class WorkflowTests(unittest.TestCase):
    def test_prepare_local_cloud_roundtrip_uses_snapshot_and_mapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/'run'; seed = run/'seed_42'
            (seed/'qtabgan').mkdir(parents=True)
            pre = TabularPreprocessor(['x'],['category'],noise_std=0)
            x = pre.fit_transform(pd.DataFrame({'x':[0.,1.,2.,3.],'category':['a','b','a','b']}))
            joblib.dump(pre,seed/'preprocessor.joblib')
            config=QTabGANConfig(n_qubits=3,n_layers=2,mapper_hidden=(8,),discriminator_hidden=(8,),
                                 hadamards_per_layer=True)
            model=QTabGAN(x.shape[1],2,config)
            torch.save({'config':asdict(config),'state_dict':model.state_dict(),'epochs_trained':0}, seed/'qtabgan/model.pt')
            write_json(run/'manifest.json',{'labels':['negative','positive'],
                                           'arguments':{'data':'fixture.csv','target':'label'}})
            args=argparse.Namespace(run=run,seed=42,rows=2,shots=128,backend='simulator',
                                    class_indices=[1,0],latent_seed=7,debias=False,output=root/'bundle')
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No network allowed')):
                prepare(args)
                local(args.output)
                plan,state=load_bundle(args.output)
                self.assertTrue(all(j['status']=='pending' for j in state['jobs']))
                service=MemoryIonQ()
                submit_bundle(args.output,service)
                probabilities=collect_bundle(args.output,service)
                np.testing.assert_allclose(probabilities,plan['ideal_probabilities'],atol=1e-12)
                map_probabilities(args.output,plan,probabilities,'ionq')
                submit_bundle(args.output,service)
                self.assertEqual(service.submissions,2)
            ideal=pd.read_csv(args.output/'local_ideal_synthetic.csv')
            remote=pd.read_csv(args.output/'ionq_synthetic.csv')
            pd.testing.assert_frame_equal(remote,ideal)
            self.assertEqual(remote.label.tolist(),['positive','negative'])
            # The snapshot makes subsequent generation independent of source files.
            (seed/'qtabgan/model.pt').unlink()
            (seed/'preprocessor.joblib').unlink()
            local(args.output)
            with self.assertRaises(FileExistsError): prepare(args)


if __name__=='__main__': unittest.main()
