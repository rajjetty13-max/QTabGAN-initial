"""Report a one-factor paired experiment with utility and raw-data diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--factor',choices=['quantum_init_gain','lr_generator','lr_discriminator'],required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();roots=sorted((a.study/'runs').glob('*/seed_*'))
    if not roots:p.error('No source runs')
    manifests=[read(root/'manifest.json') for root in roots]
    if hashlib.sha256(a.data.read_bytes()).hexdigest()!=manifests[0]['data_sha256']:p.error('Data mismatch')
    fixed=['data_sha256','ansatz','hadamards_per_layer','condition_discriminator','n_layers','target','categorical',
           'batch_size','eval_every','protocol','source_sha256','versions','quantum_init_gain','lr_generator','lr_discriminator']
    splits={};initialization={};seeds={};seen=set();budget=set()
    raw=pd.read_csv(a.data);rows=[];curves=[];losses=[];diversity=[]
    for root,m in zip(roots,manifests):
        for key in fixed:
            if key!=a.factor and m[key]!=manifests[0][key]:p.error(f'Unexpected change in {key}')
        seed=m['seed'];factor=m[a.factor]
        if (seed,factor) in seen:p.error('Repeated seed/factor')
        seen.add((seed,factor));seeds.setdefault(factor,set()).add(seed)
        protocol=(m['training_seed'],m['classical_init_seed'])
        if seed in initialization and initialization[seed]!=protocol:p.error('Initialization/training RNG differs')
        initialization[seed]=protocol
        s=np.load(root/'split.npz')
        if seed in splits:
            for key in ['train','validation','test']:
                if not np.array_equal(s[key],splits[seed][key]):p.error('Unmatched split')
        else:splits[seed]={key:s[key] for key in s.files}
        report=read(root/'final_report.json');budget.add(report['epochs'])
        curve=pd.read_csv(root/'validation_scores.csv');curves.append(curve.assign(seed=seed,factor=factor))
        losses.append(pd.read_csv(root/'losses.csv').assign(seed=seed,factor=factor))
        for epoch in curve.epoch:
            folder=root/'validation'/f'epoch_{epoch}'
            num=pd.read_csv(folder/'raw_numeric_diagnostics.csv');cat=pd.read_csv(folder/'category_coverage.csv')
            diversity.append({'seed':seed,'factor':factor,'epoch':int(epoch),
                'within_class_std_ratio':float(num.min_within_class_std_ratio.mean()),'category_coverage':float(cat.training_category_coverage.mean())})
        pre=joblib.load(root/'preprocessor.joblib');real=raw.iloc[s['train']]
        for checkpoint,key in [('best','validation_selected'),('last','last')]:
            epoch=report['best_epoch'] if checkpoint=='best' else report['epochs']
            v=curve[curve.epoch==epoch].iloc[0];folder=root/'test'/checkpoint
            fake=pd.read_csv(folder/'synthetic.csv');num=pd.read_csv(folder/'raw_numeric_diagnostics.csv');cat=pd.read_csv(folder/'category_coverage.csv')
            rows.append({'seed':seed,'factor':factor,'checkpoint':checkpoint,'epoch':epoch,
                'validation_macro_f1':float(v.f1_macro),**{k:v for k,v in report[key].items() if k!='confusion_matrix'},
                'within_class_std_ratio':float(num.min_within_class_std_ratio.mean()),'category_coverage':float(cat.training_category_coverage.mean()),
                'mean_raw_numeric_ks':float(np.mean([ks_2samp(real[c],fake[c]).statistic for c in pre.numerical_columns]))})
    if len(budget)!=1:p.error('Epoch budgets differ')
    if any(s!=next(iter(seeds.values())) for s in seeds.values()):p.error('Unpaired seed sets')
    if a.output.exists():p.error('Choose a fresh analysis folder')
    out=a.output;out.mkdir(parents=True)
    final=pd.DataFrame(rows);curves=pd.concat(curves);losses=pd.concat(losses);diversity=pd.DataFrame(diversity)
    for name,f in [('final_scores',final),('validation_scores',curves),('losses',losses),('diversity',diversity)]:f.to_csv(out/f'{name}.csv',index=False)
    metrics=['validation_macro_f1','accuracy','f1_macro','roc_auc','average_precision','within_class_std_ratio','category_coverage','mean_raw_numeric_ks']
    aggregate=final.groupby(['factor','checkpoint'])[metrics].agg(['mean','std']);aggregate.to_csv(out/'aggregate.csv')
    def trajectory(ax,frame,metric):
        for factor,part in frame.groupby('factor'):
            g=part.groupby('epoch')[metric];mean=g.mean();sd=g.std().fillna(0)
            ax.plot(mean.index,mean,marker='.',label=f'{a.factor}={factor:g}')
            ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.15)
        ax.set(xlabel='Epoch',ylabel=metric);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,metric in zip(axes.flat,['accuracy','f1_macro','roc_auc','average_precision']):trajectory(ax,curves,metric);ax.set_ylim(0,1)
    truth=pd.read_csv(roots[0]/'validation/epoch_1/predictions.csv').truth
    axes[0,0].axhline(truth.value_counts(normalize=True).max(),color='red',linestyle='--')
    axes[1,0].axhline(.5,color='gray',linestyle='--')
    fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['generator_loss','discriminator_loss','quantum_gradient_norm']):trajectory(ax,losses,metric)
    fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,metric in zip(axes,['within_class_std_ratio','category_coverage']):trajectory(ax,diversity,metric)
    for factor,part in final[final.checkpoint=='best'].groupby('factor'):
        axes[2].scatter(part.mean_raw_numeric_ks,part.f1_macro,label=f'{a.factor}={factor:g}')
    axes[2].set(xlabel='Raw numeric KS (lower is better)',ylabel='Test macro F1');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'diversity_checks.png',dpi=150);plt.close(fig)
    lines=['# Paired initialization experiment','',f'Factor: `{a.factor}`. All other recorded model/training settings are held fixed; paired splits and initialization/training seeds were checked. {next(iter(budget))} epochs per run.','',
        'Means ± sample SD. Best checkpoints use validation macro F1; last is the shared final epoch. These are exploratory reused-split results, not independent confirmation.','',
        '| Factor | Checkpoint | Validation macro F1 | Test accuracy | Test macro F1 | Test AUROC | AP | Within-class std ratio | Raw numeric KS |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (factor,checkpoint),part in final.groupby(['factor','checkpoint']):
        cols=['validation_macro_f1','accuracy','f1_macro','roc_auc','average_precision','within_class_std_ratio','mean_raw_numeric_ks']
        lines.append('| '+' | '.join([f'{factor:g}',checkpoint,*[f'{part[c].mean():.4f} ± {part[c].std():.4f}' for c in cols]])+' |')
    lines+=['','[Validation curves](validation_curves.png) · [Loss curves](loss_curves.png) · [Diversity/distribution checks](diversity_checks.png) · [Per-seed scores](final_scores.csv)','',
        'The variation ratio averages the smaller class-specific standard-deviation ratio per raw numeric feature. It is not a guarantee about every class-feature pair. KS compares original numeric distributions, not a sparse one-hot average. No model is declared successful from accuracy alone.','',
        'The quantum circuit, probability input, mapper architecture and adversarial objective are unchanged between variants. Gain changes only initialization and is folded into the first Linear layer coefficients. It does not add an inference-time normalization or replace probabilities by non-probability inputs. This initialization is an implementation experiment, not an author-confirmed setting.','',
        'The current final-H readout is still an explicit circuit extension; analytic training remains different from the paper’s sampled estimator. No hardware/noise simulation occurred. Source runs:']+[f'- `{root}`' for root in roots]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(aggregate.to_string())


if __name__=='__main__':main()
