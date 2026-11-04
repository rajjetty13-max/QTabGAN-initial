"""Summarize paired discriminator learning-rate trials without mixing protocols."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    roots=sorted((a.study/'runs').glob('*/seed_*'))
    if not roots:p.error('No trials found')
    manifests=[read(root/'manifest.json') for root in roots]
    paired={};protocols={};settings={};rows=[];curves=[];history=[];diversity=[]
    for root,m in zip(roots,manifests):
        for key in ['data_sha256','ansatz','hadamards_per_layer','condition_discriminator','n_layers','target','categorical','batch_size','eval_every','protocol','source_sha256','versions']:
            if m[key]!=manifests[0][key]:p.error(f'Unmatched {key}')
        key=(m['seed'],m['lr_generator'],m['lr_discriminator'])
        if key in paired:p.error('Duplicate seed/rate')
        split=np.load(root/'split.npz')
        if m['seed'] in paired:
            for part in ['train','validation','test']:
                if not np.array_equal(split[part],paired[m['seed']][part]):p.error('Paired split differs')
        else:paired[m['seed']]={part:split[part] for part in split.files}
        paired[key]=True
        init=(m['classical_init_seed'],m['training_seed'])
        if m['seed'] in protocols and init!=protocols[m['seed']]:p.error('Initialization/training RNG differs')
        protocols[m['seed']]=init
        variant=root.parent.name
        settings.setdefault(variant,(m['lr_generator'],m['lr_discriminator']))
        if settings[variant]!=(m['lr_generator'],m['lr_discriminator']):p.error('Inconsistent rates within a variant')
        rate=m['lr_discriminator'];seed=m['seed'];r=read(root/'final_report.json')
        scores=pd.read_csv(root/'validation_scores.csv');curves.append(scores.assign(seed=seed,variant=variant,lr_generator=m['lr_generator'],lr_discriminator=rate))
        history.append(pd.read_csv(root/'losses.csv').assign(seed=seed,variant=variant,lr_generator=m['lr_generator'],lr_discriminator=rate))
        for epoch in scores.epoch:
            folder=root/'validation'/f'epoch_{epoch}'
            n=pd.read_csv(folder/'raw_numeric_diagnostics.csv');c=pd.read_csv(folder/'category_coverage.csv')
            diversity.append({'seed':seed,'variant':variant,'lr_generator':m['lr_generator'],'lr_discriminator':rate,'epoch':int(epoch),
                              'within_class_std_ratio':float(n.min_within_class_std_ratio.mean()),
                              'category_coverage':float(c.training_category_coverage.mean())})
        for checkpoint,name in [('best','validation_selected'),('last','last')]:
            epoch=r['best_epoch'] if checkpoint=='best' else r['epochs']
            validation=scores[scores.epoch==epoch].iloc[0]
            n=pd.read_csv(root/'test'/checkpoint/'raw_numeric_diagnostics.csv')
            rows.append({'seed':seed,'variant':variant,'lr_generator':m['lr_generator'],'lr_discriminator':rate,'checkpoint':checkpoint,'epoch':epoch,
                         'validation_macro_f1':float(validation.f1_macro),
                         **{k:v for k,v in r[name].items() if k!='confusion_matrix'},
                         'within_class_std_ratio':float(n.min_within_class_std_ratio.mean())})
    budgets={read(r/'final_report.json')['epochs'] for r in roots}
    if len(budgets)!=1:p.error('Epoch budgets differ')
    per_rate={}
    for root,m in zip(roots,manifests):per_rate.setdefault(root.parent.name,set()).add(m['seed'])
    if any(s!=next(iter(per_rate.values())) for s in per_rate.values()):p.error('Seed sets differ')
    if 'control' not in settings:p.error('A control group is required')
    for rates in settings.values():
        if sum(x!=y for x,y in zip(rates,settings['control']))>1:p.error('Candidates must change only one rate')
    if a.output.exists():p.error('Choose a fresh analysis folder')
    a.output.mkdir(parents=True)
    final=pd.DataFrame(rows);curves=pd.concat(curves);history=pd.concat(history);diversity=pd.DataFrame(diversity)
    for name,frame in [('final_scores',final),('validation_scores',curves),('losses',history),('diversity',diversity)]:frame.to_csv(a.output/f'{name}.csv',index=False)
    aggregate=final.groupby(['variant','checkpoint'])[['validation_macro_f1','accuracy','f1_macro','roc_auc','average_precision','within_class_std_ratio']].agg(['mean','std'])
    aggregate.to_csv(a.output/'aggregate.csv')
    def trajectory(ax,frame,metric):
        for name,part in frame.groupby('variant'):
            grouped=part.groupby('epoch')[metric];mean=grouped.mean();sd=grouped.std().fillna(0)
            ax.plot(mean.index,mean,marker='.',label=name)
            ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.15)
        ax.set(xlabel='Epoch',ylabel=metric);ax.grid(alpha=.2);ax.legend()
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['accuracy','f1_macro','roc_auc','average_precision']):trajectory(ax,curves,metric);ax.set_ylim(0,1)
    truth=pd.read_csv(roots[0]/'validation/epoch_1/predictions.csv').truth
    axes[0,0].axhline(truth.value_counts(normalize=True).max(),color='red',linestyle='--',label='Majority reference');axes[0,0].legend()
    axes[1,0].axhline(.5,color='gray',linestyle='--')
    axes[1,1].axhline((truth==1).mean(),color='gray',linestyle='--')
    fig.suptitle('Validation performance: mean ± sample SD over paired seeds')
    fig.tight_layout();fig.savefig(a.output/'validation_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['generator_loss','discriminator_loss','quantum_gradient_norm']):trajectory(ax,history,metric)
    fig.tight_layout();fig.savefig(a.output/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['within_class_std_ratio','category_coverage']):trajectory(ax,diversity,metric)
    fig.tight_layout();fig.savefig(a.output/'diversity_curves.png',dpi=150);plt.close(fig)
    lines=['# Generator/discriminator learning-rate balance','',
        f'Each candidate changes only one learning rate from the control. Same two-layer readout circuit, mapper, D(x,y), objective, split and starting weights within paired seeds. All runs completed {next(iter(budgets))} epochs.','',
        'The supplied paper leaves rates unspecified. This is tuning an implementation choice, not recovering an author-confirmed setting. No new objective or output activation is added. The existing final-H readout remains an explicit paper extension.','',
        'Checkpoints are selected by validation macro F1. Test results are reported after selection, not used for it. Means ± sample SD over the paired seeds; short reused-split runs are exploratory, not independent confirmation.','',
        '| Variant | G rate | D rate | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | Test AP |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for (name,checkpoint),part in final.groupby(['variant','checkpoint']):
        metrics=['validation_macro_f1','accuracy','f1_macro','roc_auc','average_precision']
        cells=[f'{part[k].mean():.4f} ± {part[k].std():.4f}' for k in metrics]
        lines.append('| '+' | '.join([name,f'{settings[name][0]:g}',f'{settings[name][1]:g}',checkpoint,*cells])+' |')
    lines+=['','[Validation curves](validation_curves.png) · [Loss curves](loss_curves.png) · [Diversity curves](diversity_curves.png) · [All per-seed scores](final_scores.csv)','',
        'A discriminator near perfect real/fake accuracy establishes an easy discrimination task, not a proof that a slower learning rate will fix collapse. Non-saturating generator BCE also does not imply its logit gradient vanishes when D is confident. The intervention must be judged empirically using both utility and diversity.','',
        'Original audit: `results/training_balance_audit/`. Source runs:']+[f'- `{r}`' for r in roots]
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(aggregate.to_string())


if __name__=='__main__':main()
