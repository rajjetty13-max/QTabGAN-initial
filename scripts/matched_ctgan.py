"""Train SDV CTGAN on the exact split and encoding of an extended QTabGAN run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import time
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ctgan import CTGAN
from benchmark import evaluate, distribution, save_json
from qtabgan.evaluation import average_categorical_jsd


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.epochs<1:p.error('epochs must be positive')
    if a.output.exists():p.error('Choose a fresh output folder')
    manifest=json.loads((a.reference_run/'manifest.json').read_text())
    if hashlib.sha256(a.data.read_bytes()).hexdigest()!=manifest['data_sha256']:p.error('Data hash mismatch')
    split=np.load(a.reference_run/'split.npz');tr,va,te=[split[k] for k in ['train','validation','test']]
    df=pd.read_csv(a.data);target=manifest['target'];categorical=manifest['categorical']
    y,labels=pd.factorize(df[target],sort=True)
    pre=joblib.load(a.reference_run/'preprocessor.joblib')
    table=df.iloc[tr].copy();table[target]=y[tr]
    a.output.mkdir(parents=True)
    np.savez(a.output/'split.npz',train=tr,validation=va,test=te)
    joblib.dump(pre,a.output/'preprocessor.joblib')
    import importlib.metadata
    save_json(a.output/'manifest.json',{'model':'SDV CTGAN','ctgan_version':importlib.metadata.version('ctgan'),
        'reference_run':str(a.reference_run),'reference_manifest':manifest,'epochs':a.epochs,
        'batch_size':manifest['batch_size'],'pac':2,'generator_dim':[128,128],'discriminator_dim':[128,64,32],
        'selection':'Final epoch only; no validation checkpoint selection',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'paper_reproduction':False,'note':'CTGAN differs from paper CTAB-GAN/CTAB-GAN+; equal epochs do not imply equal compute or optimization.'})
    torch.set_num_threads(1)
    model=CTGAN(epochs=a.epochs,batch_size=manifest['batch_size'],pac=2,enable_gpu=False,
                generator_dim=(128,128),discriminator_dim=(128,64,32),verbose=True)
    model.set_random_state(manifest['seed'])
    started=time.monotonic();model.fit(table,discrete_columns=categorical+[target])
    training_seconds=time.monotonic()-started
    model.loss_values.to_csv(a.output/'losses.csv',index=False)
    fig,ax=plt.subplots()
    for name in ['Generator Loss','Discriminator Loss']:ax.plot(model.loss_values.Epoch,model.loss_values[name],label=name)
    ax.set(xlabel='CTGAN epoch',ylabel='CTGAN training loss');ax.legend();fig.tight_layout();fig.savefig(a.output/'losses.png');plt.close(fig)
    model.save(str(a.output/'model.pkl'))
    synthetic=model.sample(len(tr));synthetic_y=synthetic[target].to_numpy(dtype=int)
    if not np.isin(synthetic_y,np.arange(len(labels))).all():raise ValueError('Unexpected generated labels')
    features=synthetic.drop(columns=target);encoded=pre.transform(features)
    synthetic[target]=np.asarray(labels)[synthetic_y];synthetic.to_csv(a.output/'synthetic.csv',index=False)
    result={'epochs':a.epochs,'training_seconds':training_seconds}
    for name,indices in [('validation',va),('test',te)]:
        folder=a.output/name;folder.mkdir()
        result[name]=evaluate(encoded,synthetic_y,pre.transform(df.iloc[indices]),y[indices],folder,labels,manifest['seed'])
    folder=a.output/'distribution';folder.mkdir()
    result['encoded_fidelity']=distribution(pre.transform(df.iloc[tr]),encoded,pre.output_columns,folder)
    raw=a.output/'raw_distribution';raw.mkdir()
    result['raw_numeric_fidelity']=distribution(df.iloc[tr][pre.numerical_columns].to_numpy(),features[pre.numerical_columns].to_numpy(),pre.numerical_columns,raw)
    result['categorical_jsd']=average_categorical_jsd(df.iloc[tr][categorical].astype(str).to_numpy(),features[categorical].astype(str).to_numpy(),range(len(categorical)))
    rows=[]
    for c in pre.numerical_columns:
        for label in range(len(labels)):
            real=df.iloc[tr][c].to_numpy()[y[tr]==label];fake=features[c].to_numpy()[synthetic_y==label]
            rows.append({'feature':c,'class_index':label,'synthetic_count':len(fake),
                         'std_ratio':float(fake.std()/real.std()) if len(fake)>1 and real.std()>0 else None})
    pd.DataFrame(rows).to_csv(raw/'within_class_diversity.csv',index=False)
    save_json(a.output/'final_report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
