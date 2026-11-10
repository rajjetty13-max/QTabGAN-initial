"""Inspect D confidence and mapper variation before changing training settings."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
from qtabgan import QTabGAN,QTabGANConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a fresh output folder')
    # Reject unsupported heads before creating a misleading partial audit.
    for root in a.runs:
        for which in ['best','latest']:
            state=torch.load(root/f'{which}.pt',weights_only=True,map_location='cpu')
            if state['config'].get('tabular_output',False):
                p.error('Raw mapper diagnostics assume a linear head; mixed-type output is unsupported')
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    data=pd.read_csv(a.data);rows=[]
    for root in a.runs:
        m=json.loads((root/'manifest.json').read_text())
        import hashlib
        if hashlib.sha256(a.data.read_bytes()).hexdigest()!=m['data_sha256']:p.error('Data hash mismatch')
        target=m['target'];y,_=pd.factorize(data[target],sort=True)
        split=np.load(root/'split.npz');pre=joblib.load(root/'preprocessor.joblib')
        for which in ['best','latest']:
            s=torch.load(root/f'{which}.pt',weights_only=True,map_location='cpu')
            cfg=QTabGANConfig(**s['config']);cfg.device='cpu';cfg.diff_method='backprop';cfg.finite_shot_training=False
            model=QTabGAN(s['data_dim'],s['condition_dim'],cfg);model.load_state_dict(s['state_dict']);model.eval()
            rng=np.random.default_rng(314159)
            idx=rng.choice(split['train'],size=min(512,len(split['train'])),replace=False)
            va=rng.choice(split['validation'],size=min(512,len(split['validation'])),replace=False)
            x=torch.tensor(pre.transform(data.iloc[idx]),dtype=torch.float32)
            xv=torch.tensor(pre.transform(data.iloc[va]),dtype=torch.float32)
            c=torch.tensor(np.eye(model.condition_dim,dtype=np.float32)[y[idx]])
            cv=torch.tensor(np.eye(model.condition_dim,dtype=np.float32)[y[va]])
            z=torch.tensor(rng.uniform(0,np.pi,size=(len(idx),cfg.n_qubits)),dtype=torch.float32)
            with torch.no_grad():
                probs=model.quantum_generator(z);fake=model.mapper(probs,c)
                real_p=torch.sigmoid(model.discriminate(x,c)).flatten()
                valid_p=torch.sigmoid(model.discriminate(xv,cv)).flatten()
                fake_p=torch.sigmoid(model.discriminate(fake,c)).flatten()
                fixed_c=torch.zeros_like(c);fixed_c[:,0]=1
                out=model.mapper(probs,fixed_c)
                constant=model.mapper(probs.mean(0).expand_as(probs),fixed_c)
                first=model.mapper.model.network[0]
                quantum_term=probs@first.weight[:,:probs.shape[1]].T
                condition_term=c@first.weight[:,probs.shape[1]:].T
                rows.append({'run':str(root),'checkpoint':which,'epoch':s['epochs_trained'],
                    'D_real_train_probability':float(real_p.mean()),'D_real_validation_probability':float(valid_p.mean()),
                    'D_fake_probability':float(fake_p.mean()),
                    'D_balanced_train_accuracy':float(((real_p>=.5).float().mean()+(fake_p<.5).float().mean())/2),
                    'quantum_probability_rms_std':float(probs.std(0,unbiased=False).square().mean().sqrt()),
                    'fixed_label_generated_rms_std':float(out.std(0,unbiased=False).square().mean().sqrt()),
                    'constant_probability_ablation_rms_change':float((out-constant).square().mean().sqrt()),
                    'first_layer_quantum_contribution_rms':float(quantum_term.square().mean().sqrt()),
                    'first_layer_condition_contribution_rms':float(condition_term.square().mean().sqrt())})
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,4))
    frame.groupby('checkpoint')[['D_real_train_probability','D_real_validation_probability','D_fake_probability']].mean().plot.bar(ax=ax,rot=0)
    ax.set(ylabel='Mean probability assigned real',ylim=(0,1),title='Discriminator confidence: selected versus final state')
    fig.tight_layout();fig.savefig(a.output/'discriminator_confidence.png');plt.close(fig)
    (a.output/'README.md').write_text('# Training-balance audit\n\nThese inspect frozen saved models; no training was performed. No test rows were used. A large real/fake separation shows an easy discriminator task, not proof that changing its learning rate will help. Quantum and mapped RMS values use different coordinate systems and are not information-retention percentages.\n\nSee audit.csv and discriminator_confidence.png.\n')
    print(frame.to_string(index=False))


if __name__=='__main__':main()
