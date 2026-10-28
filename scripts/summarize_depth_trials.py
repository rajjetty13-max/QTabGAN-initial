"""Audit paired quantum-depth trials and report validation, utility and diversity."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import pennylane as qml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qtabgan.quantum_generator import QuantumGeneratorCore


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ms=[read(root/'manifest.json') for root in a.runs]
    if len({(m['seed'],m['n_layers']) for m in ms})!=len(ms):p.error('Duplicate depth and seed')
    expected_seeds={m['seed'] for m in ms};layers=sorted({m['n_layers'] for m in ms})
    for layer in layers:
        if {m['seed'] for m in ms if m['n_layers']==layer}!=expected_seeds:p.error('Unpaired seed sets')
    protocols={};splits={}
    for root,m in zip(a.runs,ms):
        for key in ['data_sha256','ansatz','hadamards_per_layer','condition_discriminator','target','categorical','batch_size','eval_every','protocol','source_sha256','versions']:
            if m[key]!=ms[0][key]:p.error(f'Incompatible {key}')
        if m['classical_init_seed'] is None:p.error('Independent classical initialization required')
        protocol=(m['classical_init_seed'],m['training_seed'])
        if m['seed'] in protocols and protocol!=protocols[m['seed']]:p.error('Paired initialization/training seeds differ')
        protocols[m['seed']]=protocol
        split=np.load(root/'split.npz')
        if m['seed'] in splits:
            for key in ['train','validation','test']:
                if not np.array_equal(split[key],splits[m['seed']][key]):p.error('Paired split mismatch')
        else:splits[m['seed']]={k:split[k] for k in split.files}
    reports=[read(root/'final_report.json') for root in a.runs]
    if len({r['epochs'] for r in reports})!=1:p.error('Epoch budgets differ')
    if a.output.exists():p.error('Choose a fresh output folder')
    out=a.output;out.mkdir(parents=True)
    rows=[];curves=[];losses=[];diversity=[];resources=[]
    for root,m,r in zip(a.runs,ms,reports):
        l=m['n_layers'];seed=m['seed']
        v=pd.read_csv(root/'validation_scores.csv');curves.append(v.assign(layers=l,seed=seed))
        h=pd.read_csv(root/'losses.csv').assign(layers=l,seed=seed)
        h['quantum_gradient_rms']=h.quantum_gradient_norm/np.sqrt(16*l);losses.append(h)
        for epoch in v.epoch:
            folder=root/'validation'/f'epoch_{epoch}'
            numeric=pd.read_csv(folder/'raw_numeric_diagnostics.csv');cat=pd.read_csv(folder/'category_coverage.csv')
            diversity.append({'layers':l,'seed':seed,'epoch':int(epoch),
                'within_class_std_ratio':float(numeric.min_within_class_std_ratio.mean()),
                'category_coverage':float(cat.training_category_coverage.mean())})
        for checkpoint,key in [('best','validation_selected'),('last','last')]:
            epoch=r['best_epoch'] if checkpoint=='best' else r['epochs']
            selection=v[v.epoch==epoch].iloc[0]
            folder=root/'test'/checkpoint
            numeric=pd.read_csv(folder/'raw_numeric_diagnostics.csv')
            rows.append({'layers':l,'seed':seed,'checkpoint':checkpoint,'epoch':epoch,
                'validation_f1_macro':float(selection.f1_macro),
                **{k:value for k,value in r[key].items() if k!='confusion_matrix'},
                'within_class_std_ratio':float(numeric.min_within_class_std_ratio.mean())})
        if l not in [d['layers'] for d in resources]:
            saved=torch.load(root/'latest.pt',weights_only=True,map_location='cpu')
            core=QuantumGeneratorCore(8,l,ansatz=m['ansatz'],hadamards_per_layer=m['hadamards_per_layer'])
            core.load_state_dict({'weights':saved['state_dict']['quantum_generator.weights']})
            z=torch.zeros(8);spec=qml.specs(core.circuit)(z,core.weights)['resources']
            resources.append({'layers':l,'parameters':core.weights.numel(),'gates':spec.num_gates,'logical_depth':spec.depth,
                              'cnot_gates':sum(g.name=='CNOT' for g in qml.tape.make_qscript(core._probabilities)(z,core.weights).operations)})
            (out/f'circuit_{l}_layers.txt').write_text(qml.draw(core.circuit)(z,core.weights))
            fig,_=qml.draw_mpl(core.circuit)(z,core.weights);fig.savefig(out/f'circuit_{l}_layers.png',bbox_inches='tight');plt.close(fig)
    final=pd.DataFrame(rows);curves=pd.concat(curves);losses=pd.concat(losses);diversity=pd.DataFrame(diversity)
    for name,f in [('final_scores',final),('validation_scores',curves),('losses',losses),('diversity',diversity),('resources',pd.DataFrame(resources))]:
        f.to_csv(out/f'{name}.csv',index=False)
    def trajectory(ax,frame,metric):
        for layer,part in frame.groupby('layers'):
            grouped=part.groupby('epoch')[metric];mean=grouped.mean();sd=grouped.std().fillna(0)
            ax.plot(mean.index,mean,marker='.',label=f'{layer} layers')
            ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.15)
        ax.set(xlabel='Epoch',ylabel=metric);ax.legend();ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['accuracy','f1_macro','roc_auc','average_precision']):
        trajectory(ax,curves,metric);ax.set_ylim(0,1)
    majority=np.mean([pd.read_csv(root/'validation/epoch_1/predictions.csv').truth.value_counts(normalize=True).max() for root in a.runs])
    axes[0,0].axhline(majority,color='red',linestyle='--',label=f'Majority reference {majority:.1%}')
    axes[0,0].legend()
    axes[1,0].axhline(.5,color='gray',linestyle='--')
    fig.suptitle('Validation means ± sample SD; same paired seed sets')
    fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['generator_loss','discriminator_loss','quantum_gradient_rms']):trajectory(ax,losses,metric)
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['within_class_std_ratio','category_coverage']):trajectory(ax,diversity,metric)
    fig.tight_layout();fig.savefig(out/'diversity_curves.png',dpi=150);plt.close(fig)
    aggregate=final.groupby(['layers','checkpoint'])[['accuracy','f1_macro','roc_auc','average_precision','within_class_std_ratio']].agg(['mean','std'])
    aggregate.to_csv(out/'aggregate.csv')
    validation_aggregate=curves.groupby(['layers','epoch'])[['accuracy','f1_macro','roc_auc','average_precision']].agg(['mean','std'])
    validation_aggregate.to_csv(out/'validation_aggregate.csv')
    lines=['# Circuit depth comparison','',
        f'Paper-based readout circuit, {len(expected_seeds)} paired seeds, {reports[0]["epochs"]} epochs. Only quantum layer count differs. Independent classical initialization and training RNG are held fixed within each seed; original quantum angle prefixes also match. Equal epochs do not imply equal gate execution or compute.','',
        'The readout Hadamards are an explicit extension to the paper. Variational layers retain its RY/RZ operations and ordered CNOT ring. The paper leaves depth unspecified; extra layers test that choice without adding a classical model or objective.','',
        '## Test scores','', f'Mean ± sample SD. “best” uses maximum validation macro F1 over the recorded checkpoints. “last” uses epoch {reports[0]["epochs"]}. Test scores are reported after training, not used to choose checkpoints.','',
        '| Layers | Checkpoint | Accuracy | Macro F1 | AUROC | AP |','| ---: | --- | ---: | ---: | ---: | ---: |']
    for (layer,checkpoint),part in final.groupby(['layers','checkpoint']):
        cells=[f'{part[k].mean():.4f} ± {part[k].std():.4f}' for k in ['accuracy','f1_macro','roc_auc','average_precision']]
        lines.append('| '+' | '.join([str(layer),checkpoint,*cells])+' |')
    lines+=['','## Circuit cost','', '| Layers | Parameters | Gates | CNOTs | Logical depth |','| ---: | ---: | ---: | ---: | ---: |']
    for r in resources:lines.append(f'| {r["layers"]} | {r["parameters"]} | {r["gates"]} | {r["cnot_gates"]} | {r["logical_depth"]} |')
    lines+=['','## Evidence','', '[Validation curves](validation_curves.png) · [Loss/gradient curves](loss_curves.png) · [Diversity curves](diversity_curves.png) · [Per-seed scores](final_scores.csv) · [Aggregate](aggregate.csv)','',
        'The gradient plot uses L2 gradient norm divided by the square root of the number of angles, to avoid confusing a larger parameter count with stronger per-angle gradients. Within-class variation averages the smaller class-specific standard-deviation ratio per numeric feature. Small values still indicate collapse.','',
        'This experiment uses a fresh classical initialization protocol, so previous result tables are not its matched controls. Previously examined Loan splits are reused: the test set is not independent confirmation. Three seeds provide exploratory evidence, not statistical significance or verified reproduction. No noise/hardware experiment was performed.','',
        'Reproduce with `extended_benchmark.py --ansatz paper_readout --condition-discriminator --layers L --classical-init-seed C`, retaining all other recorded arguments and using a fresh output folder. Source runs:']+[f'- `{r}`' for r in a.runs]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(aggregate.to_string())


if __name__=='__main__':main()
