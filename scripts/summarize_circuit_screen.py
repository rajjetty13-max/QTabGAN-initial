"""Aggregate matched circuit-only screens; candidate choice uses validation macro F1."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())


def variant(m):
    if m['ansatz']=='paper_readout':return 'Repeated H + readout H' if m['hadamards_per_layer'] else 'Readout H'
    return 'Repeated H' if m['hadamards_per_layer'] else 'Original'


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifests=[read(root/'manifest.json') for root in a.runs]
    if len({(m['seed'],variant(m)) for m in manifests})!=len(manifests):p.error('Duplicate variant/seed')
    for m in manifests:
        if m.get('n_layers',2)!=2:p.error('Use the depth report for different layer counts')
        if m['ansatz'] not in ['paper','paper_readout']:p.error('This screen compares paper blocks and Hadamard changes only')
        for key in ['data_sha256','condition_discriminator','batch_size','eval_every','target','categorical','protocol','source_sha256','versions']:
            if m[key]!=manifests[0][key]:p.error(f'Runs differ in {key}')
        if m['training_seed']!=m['seed']+20000:p.error('Unexpected training RNG seed')
    if hashlib.sha256(a.data.read_bytes()).hexdigest()!=manifests[0]['data_sha256']:p.error('Data hash mismatch')
    budgets={read(root/'final_report.json')['epochs'] for root in a.runs}
    if len(budgets)!=1:p.error('Epoch budgets differ')
    splits={}
    for root,m in zip(a.runs,manifests):
        s=np.load(root/'split.npz')
        if m['seed'] in splits:
            for key in ['train','validation','test']:
                if not np.array_equal(s[key],splits[m['seed']][key]):p.error('Paired split mismatch')
        else:splits[m['seed']]={key:s[key] for key in s.files}
    if a.output.exists():p.error('Choose a fresh output folder')
    a.output.mkdir(parents=True)
    data=pd.read_csv(a.data);rows=[];curves=[]
    for root,m in zip(a.runs,manifests):
        report=read(root/'final_report.json');v=variant(m)
        scores=pd.read_csv(root/'validation_scores.csv');curves.append(scores.assign(variant=v,seed=m['seed']))
        pre=joblib.load(root/'preprocessor.joblib');real=data.iloc[splits[m['seed']]['train']]
        for checkpoint,key in [('best','validation_selected'),('last','last')]:
            folder=root/'test'/checkpoint
            numeric=pd.read_csv(folder/'raw_numeric_diagnostics.csv');fake=pd.read_csv(folder/'synthetic.csv')
            cat=pd.read_csv(folder/'category_coverage.csv')
            epoch=report['best_epoch'] if checkpoint=='best' else report['epochs']
            val=scores[scores.epoch==epoch].iloc[0]
            rows.append({'variant':v,'seed':m['seed'],'checkpoint':checkpoint,'epoch':epoch,
                         'validation_f1_macro':float(val.f1_macro),'validation_roc_auc':float(val.roc_auc),
                         **{k:value for k,value in report[key].items() if k!='confusion_matrix'},
                         'mean_minimum_within_class_std_ratio':float(numeric.min_within_class_std_ratio.mean()),
                         'mean_category_coverage':float(cat.training_category_coverage.mean()),
                         'mean_raw_numeric_ks':float(np.mean([ks_2samp(real[c],fake[c]).statistic for c in pre.numerical_columns]))})
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'all_results.csv',index=False)
    curves=pd.concat(curves,ignore_index=True);curves.to_csv(a.output/'validation_curves.csv',index=False)
    metrics=['accuracy','f1_macro','roc_auc','average_precision','mean_raw_numeric_ks','mean_minimum_within_class_std_ratio','mean_category_coverage']
    frame.groupby(['variant','checkpoint'])[metrics].agg(['mean','std','count']).to_csv(a.output/'aggregate.csv')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,metric in zip(axes.flat,['accuracy','f1_macro','roc_auc','average_precision']):
        for (name,seed),part in curves.groupby(['variant','seed']):
            ax.plot(part.epoch,part[metric],marker='.',label=f'{name}, seed {seed}')
        ax.set(xlabel='Epoch',ylabel=f'Validation {metric}',ylim=(0,1));ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(a.output/'validation_curves.png',dpi=150);plt.close(fig)
    paired=frame[(frame.checkpoint=='best')&frame.variant.isin(['Original','Readout H'])]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['f1_macro','roc_auc','average_precision']):
        for seed,part in paired.groupby('seed'):
            pair=part.set_index('variant')
            if {'Original','Readout H'}<=set(pair.index):
                ax.plot([0,1],pair.loc[['Original','Readout H'],metric],marker='o',label=f'Seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['Original','Readout H'],ylim=(0,1),title=f'Test {metric}');ax.legend()
    fig.tight_layout();fig.savefig(a.output/'paired_test_scores.png',dpi=150);plt.close(fig)
    lines=['# Circuit-only improvement experiment','',
           'All runs use the same original paper variational blocks, eight qubits, two layers, 32 parameters, mapper, D(x,y), optimizer settings and preprocessing. Split rows and training RNG are paired within each seed. The only changes are repeated Hadamards and/or final Hadamards before measurement. Final readout is an explicit extension to the paper, not exact reproduction.','',
           'The initial seed-42 screen included four variants. Readout H was taken to seeds 43/44 because it made inactive parameters effective and essentially tied the original validation macro F1 at seed 42. This exploratory selection and reused splits limit any generalization claim. Tests did not select checkpoints: highest validation macro F1 did.','',
           '## Per-seed validation-selected checkpoints','',
           '| Variant | Seed | Epoch | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in frame[frame.checkpoint=='best'].to_dict('records'):
        lines.append(f'| {r["variant"]} | {r["seed"]} | {r["epoch"]} | {r["validation_f1_macro"]:.4f} | {r["accuracy"]:.4f} | {r["f1_macro"]:.4f} | {r["roc_auc"]:.4f} | {r["average_precision"]:.4f} |')
    lines+=['','## Matched original versus readout','', 'Means ± sample SD for each available seed set; confirm the seed counts below. “last” uses the shared final epoch.','',
            '| Variant | Checkpoint | Seeds | Accuracy | Macro F1 | AUROC | AP |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (name,checkpoint),part in frame[frame.variant.isin(['Original','Readout H'])].groupby(['variant','checkpoint']):
        cells=[f'{part[k].mean():.4f} ± {part[k].std():.4f}' for k in metrics[:4]]
        lines.append('| '+' | '.join([name,checkpoint,str(len(part)),*cells])+' |')
    lines+=['','## Artifacts','', '[Validation curves](validation_curves.png) · [Paired test scores](paired_test_scores.png) · [All numbers, including diversity](all_results.csv) · [Aggregate](aggregate.csv)','',
            'The readout circuit activates all 32 parameter slots versus 24 in the original, adds 8 single-qubit gates, retains 16 CNOTs, and changes logical depth from 22 to 23. Structural evidence is in `results/readout_circuit_diagnostics/`. This is not a noise or hardware advantage.','',
            'These are 30-epoch local analytic simulations, not an established paper benchmark. Higher utility is insufficient if diversity remains collapsed. Every source folder includes losses, raw and encoded feature diagnostics, ROC/PR curves, confusion counts, synthetic data, initial circuit diagrams and saved best/latest model states.','',
            'Runs:']+[f'- `{root}`' for root in a.runs]
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(frame[frame.checkpoint=='best'].to_string(index=False))


if __name__=='__main__':main()
