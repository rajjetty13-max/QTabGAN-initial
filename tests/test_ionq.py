"""Offline contract tests: no IonQ credentials, account or network required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import uuid

import numpy as np
import torch

from qtabgan.quantum_generator import QuantumGeneratorCore
from qtabgan.ionq import (API_URL, IonQClient, attach_job, collect_bundle, digest,
                         export_qis, load_bundle, probabilities_from_ionq,
                         simulate_qis, submit_bundle, write_json)


class IonQTests(unittest.TestCase):
    def test_export_preserves_probabilities_and_gate_order(self):
        torch.manual_seed(8)
        for repeated in [False, True]:
            core = QuantumGeneratorCore(3, 2, hadamards_per_layer=repeated).double()
            noise = torch.rand(3, dtype=torch.float64)
            circuit = export_qis(core, noise)
            expected = core.circuit(noise, core.weights).detach().numpy()
            np.testing.assert_allclose(simulate_qis(circuit), expected, atol=1e-12)
            cnots = [(g['control'],g['target']) for g in circuit['circuit'] if g['gate']=='cnot']
            self.assertEqual(cnots, [(0,1),(1,2),(2,0)] * 2)
            self.assertEqual(sum(g['gate']=='h' for g in circuit['circuit']), 6 if repeated else 3)

    def test_asymmetric_basis_state_and_decimal_bit_order(self):
        circuit = {'qubits':3,'gateset':'qis','circuit':[{'gate':'ry','rotation':np.pi,'target':0}]}
        expected = simulate_qis(circuit)
        self.assertAlmostEqual(expected[4], 1.)  # PennyLane |100>.
        decoded = probabilities_from_ionq({'1':1.}, 3)  # IonQ q0 at 2**0.
        np.testing.assert_allclose(decoded, expected, atol=1e-12)
        np.testing.assert_allclose(probabilities_from_ionq({'3':.25,'4':.75},3),
                                   [0,.75,0,0,0,0,.25,0])

    def test_reject_invalid_or_ambiguous_results(self):
        for histogram in [{}, {'8':1.}, {'01':1.}, {'0':128}, {'0':.2},
                          {'0':float('nan')}, {'0':-.1,'1':1.1},
                          {'output_all':{'00':1.}}, {'0':True}]:
            with self.subTest(histogram=histogram), self.assertRaises(ValueError):
                probabilities_from_ionq(histogram, 3)

    def make_bundle(self, directory, rows=2):
        directory = Path(directory)
        for name in ['checkpoint.pt','preprocessor.joblib']:
            (directory/name).write_bytes(b'fixture')
        plan={'schema_version':1,'n_qubits':3,
              'assets':{n:digest(directory/n) for n in ['checkpoint.pt','preprocessor.joblib']},
              'requests':[{'name':f'unique-row-{i}','backend':'simulator'} for i in range(rows)]}
        write_json(directory/'bundle.json',plan)
        write_json(directory/'state.json',{'bundle_sha256':digest(directory/'bundle.json'),
                    'jobs':[{'status':'pending','job_id':None} for _ in range(rows)]})
        return plan

    def test_submit_resume_and_collect_preserve_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_bundle(directory)
            client=Mock()
            ids=[str(uuid.uuid4()),str(uuid.uuid4())]
            client.submit.side_effect=[{'id':i} for i in ids]
            submit_bundle(directory,client)
            submit_bundle(directory,client)
            self.assertEqual(client.submit.call_count,2)
            jobs={ids[i]:{'id':ids[i], 'status':'completed', **plan['requests'][i]} for i in range(2)}
            client.job.side_effect=lambda job_id: jobs[job_id]
            client.probabilities.side_effect=lambda job_id, job: {'1':1.} if job_id==ids[0] else {'4':1.}
            probabilities=collect_bundle(directory,client)
            self.assertEqual(np.argmax(probabilities,axis=1).tolist(),[4,1])
            np.testing.assert_equal(collect_bundle(directory,client),probabilities)
            self.assertEqual(client.probabilities.call_count,2)

    def test_uncertain_submission_is_not_retried_and_can_be_attached(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_bundle(directory,rows=1)
            client=Mock(); client.submit.side_effect=TimeoutError('lost response')
            with self.assertRaises(TimeoutError): submit_bundle(directory,client)
            with self.assertRaises(RuntimeError): submit_bundle(directory,client)
            self.assertEqual(client.submit.call_count,1)
            job_id=str(uuid.uuid4())
            client.job.return_value={'id':job_id,**plan['requests'][0]}
            attach_job(directory,0,job_id,client)
            submit_bundle(directory,client)
            self.assertEqual(client.submit.call_count,1)

    def test_pending_failed_and_mismatched_jobs_never_generate_fallback_data(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=self.make_bundle(directory,rows=1)
            client=Mock(); job_id=str(uuid.uuid4()); client.submit.return_value={'id':job_id}
            submit_bundle(directory,client)
            client.job.return_value={'status':'started',**plan['requests'][0]}
            self.assertIsNone(collect_bundle(directory,client))
            client.job.return_value={'status':'failed',**plan['requests'][0]}
            self.assertIsNone(collect_bundle(directory,client))
            client.probabilities.assert_not_called()
            client.job.return_value={'status':'completed','backend':'wrong','name':'wrong'}
            with self.assertRaises(ValueError): collect_bundle(directory,client)

    def test_bundle_and_asset_integrity(self):
        for filename in ['checkpoint.pt','bundle.json']:
            with tempfile.TemporaryDirectory() as directory:
                self.make_bundle(directory)
                with (Path(directory)/filename).open('a') as handle: handle.write(' ')
                with self.assertRaises(ValueError): load_bundle(directory)

    def test_api_contract_and_credentials_are_not_in_payload(self):
        session=Mock()
        response=Mock(status_code=201); response.json.return_value={'id':'test'}
        session.request.return_value=response
        client=IonQClient(api_key='not-a-real-key',session=session)
        client.submit({'backend':'simulator'})
        args,kwargs=session.request.call_args
        self.assertEqual(args,('POST',API_URL+'/jobs'))
        self.assertEqual(kwargs['headers']['Authorization'],'apiKey not-a-real-key')
        self.assertEqual(kwargs['json'],{'backend':'simulator'})
        self.assertEqual(kwargs['timeout'],30)
        self.assertFalse(kwargs['allow_redirects'])
        session.request.assert_called_once()
        response.status_code=401
        with self.assertRaisesRegex(RuntimeError,'HTTP 401'): client.backends()

    def test_current_v1_artifact_and_flat_endpoint(self):
        session=Mock(); response=Mock(status_code=200)
        response.json.return_value={'0':1.}; session.request.return_value=response
        client=IonQClient(api_key='fixture',session=session)
        job_id,artifact_id=str(uuid.uuid4()),str(uuid.uuid4())
        client.probabilities(job_id,{'results':{'ionq.result.probabilities.json.v1':{'id':artifact_id}}})
        self.assertEqual(session.request.call_args.args[1],f'{API_URL}/jobs/{job_id}/artifacts/{artifact_id}')
        client.probabilities(job_id,{})
        self.assertEqual(session.request.call_args.args[1],f'{API_URL}/jobs/{job_id}/results/probabilities')


if __name__=='__main__': unittest.main()
