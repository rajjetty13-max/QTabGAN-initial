"""Plot validation learning curves and final test results from matched multi-seed runs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from benchmark import evaluate


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifests=[read(run/'manifest.json') for run in args.runs]
    reports=[read(run/'final_report.json') for run in args.runs]
    for manifest in manifests[1:]:
        for key in ['data_sha256','ansatz','n_layers','classical_init_seed','hadamards_per_layer','condition_discriminator','target','categorical','batch_size','eval_every','protocol','source_sha256','versions']:
            if manifest.get(key)!=manifests[0].get(key): parser.error(f'Incompatible {key}')
    if len({m['seed'] for m in manifests})!=len(manifests): parser.error('Repeated seed')
    if len({r['epochs'] for r in reports})!=1: parser.error('Epoch budgets differ')
    if hashlib.sha256(args.data.read_bytes()).hexdigest()!=manifests[0]['data_sha256']: parser.error('Data hash mismatch')
    if args.output.exists(): parser.error('Choose a fresh output folder')
    out=args.output;out.mkdir(parents=True)
    raw=pd.read_csv(args.data)
    target=manifests[0]['target'];y,labels=pd.factorize(raw[target],sort=True)
    curves=[];history=[];diversity=[];final=[];references=[]
    for run,manifest,report in zip(args.runs,manifests,reports):
        seed=manifest['seed']
        curve=pd.read_csv(run/'validation_scores.csv').assign(seed=seed);curves.append(curve)
        history.append(pd.read_csv(run/'losses.csv').assign(seed=seed))
        split=np.load(run/'split.npz');tr,va=split['train'],split['validation']
        pre=joblib.load(run/'preprocessor.joblib')
        folder=out/f'real_validation_seed_{seed}';folder.mkdir()
        baseline=evaluate(pre.transform(raw.iloc[tr]),y[tr],pre.transform(raw.iloc[va]),y[va],folder,labels,seed)
        majority=np.bincount(y[tr]).argmax()
        references.append({'seed':seed,**{k:v for k,v in baseline.items() if k!='confusion_matrix'},
                           'majority_accuracy':float((y[va]==majority).mean()),
                           'majority_macro_f1':float(f1_score(y[va],np.full(len(va),majority),average='macro')),
                           'positive_prevalence':float((y[va]==1).mean())})
        for epoch in curve.epoch:
            folder=run/'validation'/f'epoch_{epoch}'
            numeric=pd.read_csv(folder/'raw_numeric_diagnostics.csv')
            category=pd.read_csv(folder/'category_coverage.csv')
            diversity.append({'seed':seed,'epoch':int(epoch),
                              'mean_minimum_within_class_std_ratio':float(numeric.min_within_class_std_ratio.mean()),
                              'mean_numeric_std_ratio':float(numeric.std_ratio.mean()),
                              'mean_category_coverage':float(category.training_category_coverage.mean())})
        for checkpoint,key in [('best','validation_selected'),('last','last'),('real','real')]:
            final.append({'seed':seed,'checkpoint':checkpoint,
                          'epoch':report['best_epoch'] if checkpoint=='best' else report['epochs'],
                          **{k:v for k,v in report[key].items() if k!='confusion_matrix'}})
    curves=pd.concat(curves,ignore_index=True);history=pd.concat(history,ignore_index=True)
    diversity=pd.DataFrame(diversity);final=pd.DataFrame(final);references=pd.DataFrame(references)
    for name,frame in [('validation_scores',curves),('losses',history),('diversity',diversity),('final_test',final),('real_validation',references)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    summary=curves.groupby('epoch')[['accuracy','f1_macro','roc_auc','average_precision']].agg(['mean','std'])
    summary.to_csv(out/'validation_aggregate.csv')
    def trajectory(ax,frame,metric):
        for seed,part in frame.groupby('seed'):
            ax.plot(part.epoch,part[metric],alpha=.55,linewidth=1,label=f'Seed {seed}')
        grouped=frame.groupby('epoch')[metric]
        mean=grouped.mean();std=grouped.std().fillna(0)
        ax.plot(mean.index,mean,color='black',linewidth=2,label='Mean')
        ax.fill_between(mean.index,mean-std,mean+std,color='black',alpha=.1,label='±1 sample SD')
        ax.set(xlabel='Epoch',ylabel=metric);ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,metric in zip(axes.flat,['accuracy','f1_macro','roc_auc','average_precision']):
        trajectory(ax,curves,metric)
        ax.axhline(references[metric].mean(),linestyle=':',color='green',label='Real-trained validation mean')
        baseline={'accuracy':references.majority_accuracy.mean(),'f1_macro':references.majority_macro_f1.mean(),
                  'roc_auc':.5,'average_precision':references.positive_prevalence.mean()}[metric]
        ax.axhline(baseline,linestyle='--',color='red',label='Majority/chance/prevalence reference')
        ax.set_ylim(0,1);ax.legend(fontsize=7)
    fig.suptitle('Classifier trained on generated data, evaluated on real validation rows')
    fig.tight_layout();fig.savefig(out/'validation_learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));trajectory(ax,curves,'accuracy')
    ax.axhline(references.majority_accuracy.mean(),color='red',linestyle='--',label=f'Majority accuracy {references.majority_accuracy.mean():.1%}')
    ax.axhline(references.accuracy.mean(),color='green',linestyle=':',label=f'Real-trained validation {references.accuracy.mean():.1%}')
    ax.set(ylim=(0,1),title='Validation accuracy versus training epoch');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'accuracy_curve.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,['generator_loss','discriminator_loss','quantum_gradient_norm']):
        trajectory(ax,history,metric);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['mean_minimum_within_class_std_ratio','mean_category_coverage']):
        trajectory(ax,diversity,metric);ax.axhline(1,color='green',linestyle=':',label='Real diversity/full coverage reference')
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'diversity_curves.png',dpi=150);plt.close(fig)
    final.groupby('checkpoint')[['accuracy','f1_macro','roc_auc','average_precision']].agg(['mean','std']).to_csv(out/'final_test_aggregate.csv')
    lines=['# 100-epoch paper-based training experiment','',
           'Original eight-qubit/two-layer circuit, ReLU mapper with linear output, and label-conditioned D(x,y). Three full-Loan runs, seeds 42/43/44; 60/20/20 split per seed; batch size 64. Evaluation at epoch 1 and every 10 epochs. Analytic local simulation; no hardware noise.','',
           'The curves show downstream classifier utility on validation rows. They are not discriminator accuracy. Thin lines are individual seeds; the black line is the mean and shading is one sample standard deviation. All measured points are retained without smoothing.','',
           '## Validation trajectory','', '| Epoch | Accuracy | Macro F1 | AUROC | Average precision |', '| --- | ---: | ---: | ---: | ---: |']
    for epoch in [1,10,30,50,100]:
        if epoch in summary.index:
            lines.append('| '+str(epoch)+' | '+' | '.join(f'{summary.loc[epoch,(m,"mean")]:.4f}' for m in ['accuracy','f1_macro','roc_auc','average_precision'])+' |')
    lines+=['','## Final test results','', 'Checkpoints were selected by validation macro F1. “last” is epoch 100; “real” is the real-trained classifier reference. Means ± sample SD across three seeds.','',
            '| Checkpoint | Accuracy | Macro F1 | AUROC | Average precision |','| --- | ---: | ---: | ---: | ---: |']
    for checkpoint,part in final.groupby('checkpoint'):
        lines.append('| '+checkpoint+' | '+' | '.join(f'{part[m].mean():.4f} ± {part[m].std():.4f}' for m in ['accuracy','f1_macro','roc_auc','average_precision'])+' |')
    lines+=['','Selected epochs: '+', '.join(f'seed {r["seed"]}: {r["epoch"]}' for r in final[final.checkpoint=='best'].to_dict('records'))+'.','',
            '## Plots and numbers','', '[Accuracy curve](accuracy_curve.png) · [All validation metrics](validation_learning_curves.png) · [Loss curves](loss_curves.png) · [Diversity curves](diversity_curves.png)','',
            '[Per-seed final test numbers](final_test.csv) · [Validation CSV](validation_scores.csv) · [Diversity CSV](diversity.csv)','',
            'These exploratory runs reuse previously examined splits. The test results are not a new independent confirmation. Three seeds mix split and initialization variation; they do not establish statistical significance or a verified paper reproduction. No circuit variant was trained in this experiment, so the curves alone cannot attribute a failure to the quantum circuit.']
    lines+=['','Source runs:']+[f'- `{run}`' for run in args.runs]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(final.to_string(index=False))


if __name__=='__main__':main()
