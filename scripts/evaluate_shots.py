"""Paired analytic/finite-shot generation from a trusted local QTabGAN checkpoint."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
from benchmark import evaluate, distribution, save_json
from qtabgan import QTabGAN, QTabGANConfig

p = argparse.ArgumentParser(__doc__)
p.add_argument('run', type=Path)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--shots', type=int, default=1024)
a = p.parse_args(); torch.set_num_threads(1)
manifest = json.loads((a.run / 'manifest.json').read_text())
args = manifest['arguments']
if not args['data']: p.error('A CSV-backed benchmark run is required')
if hashlib.sha256(Path(args['data']).read_bytes()).hexdigest() != manifest['source_sha256']:
    p.error('Dataset hash does not match the recorded run')
frame = pd.read_csv(args['data']); target = args['target']
y, labels = pd.factorize(frame[target], sort=True)
root = a.run / f'seed_{a.seed}'
split = np.load(root / 'split.npz'); tr, te = split['train'], split['test']
pre = joblib.load(root / 'preprocessor.joblib')  # Only load trusted local artifacts.
x = pre.transform(frame.iloc[tr]); test = pre.transform(frame.iloc[te])
checkpoint = torch.load(root / 'qtabgan/model.pt', weights_only=True, map_location='cpu')
config = QTabGANConfig(**checkpoint['config']); config.shots=a.shots; config.device='cpu'
training_mode = 'finite_shot' if config.finite_shot_training else 'analytic'
config.finite_shot_training = False  # Compare analytic and sampled inference independently of training.
model = QTabGAN(x.shape[1], len(labels), config); model.load_state_dict(checkpoint['state_dict'])
condition = np.eye(len(labels), dtype=np.float32)[y[tr]]
rows=[]
for sampled in [False, True]:
    name = 'finite_shot' if sampled else 'analytic'
    out = root / 'qtabgan' / f'paired_{a.shots}_shots' / name; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed + 1000)  # Identical latent angles for each mode.
    generated = model.generate(condition, sample=sampled)
    decoded = pre.inverse_transform(generated); fake=pre.transform(decoded)
    utility = evaluate(fake, y[tr], test, y[te], out, labels, a.seed)
    fidelity = distribution(x, fake, pre.output_columns, out)
    decoded[target] = np.asarray(labels)[y[tr]]; decoded.to_csv(out/'synthetic.csv',index=False)
    metrics = {'mode':name, 'shots':a.shots if sampled else None, 'training':training_mode,
               'gate_noise':False, 'epochs_trained':checkpoint['epochs_trained'], **utility, **fidelity}
    save_json(out/'metrics.json', metrics); rows.append(metrics)
pd.DataFrame(rows).to_csv(root/'qtabgan'/f'paired_{a.shots}_shots'/'summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
