"""Compare a matched CTGAN endpoint with one extended QTabGAN run."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--qtab-run',type=Path,required=True)
    p.add_argument('--ctgan-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    qmanifest=read(a.qtab_run/'manifest.json');cmanifest=read(a.ctgan_run/'manifest.json')
    if qmanifest!=cmanifest['reference_manifest']:p.error('Reference manifests differ')
    qs=np.load(a.qtab_run/'split.npz');cs=np.load(a.ctgan_run/'split.npz')
    for key in ['train','validation','test']:
        if not np.array_equal(qs[key],cs[key]):p.error(f'{key} rows differ')
    q=read(a.qtab_run/'final_report.json');c=read(a.ctgan_run/'final_report.json')
    if q['epochs']!=c['epochs']:p.error('Final epoch budgets differ')
    if a.output.exists():p.error('Choose a fresh output folder')
    a.output.mkdir(parents=True)
    rows=[]
    for name,score in [(f'QTabGAN epoch {q["epochs"]}',q['last']),
                       (f'QTabGAN selected epoch {q["best_epoch"]}',q['validation_selected']),
                       (f'SDV CTGAN epoch {c["epochs"]}',c['test']),('Real-trained reference',q['real'])]:
        rows.append({'model':name,**{k:v for k,v in score.items() if k!='confusion_matrix'},
                     'confusion_matrix':json.dumps(score['confusion_matrix'])})
    pd.DataFrame(rows).to_csv(a.output/'summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,folder,score in [('QTabGAN final',a.qtab_run/'test/last',q['last']),('CTGAN final',a.ctgan_run/'test',c['test'])]:
        roc=pd.read_csv(folder/'roc_curve.csv');pr=pd.read_csv(folder/'precision_recall_curve.csv')
        axes[0].plot(roc.false_positive_rate,roc.true_positive_rate,label=f'{name}: {score["roc_auc"]:.3f}')
        axes[1].plot(pr.recall,pr.precision,label=f'{name}: {score["average_precision"]:.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC / AUROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Test precision–recall / AP')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(a.output/'auc_comparison.png',dpi=150);plt.close(fig)
    lines=['# Matched CTGAN versus QTabGAN','',
           f'Seed {qmanifest["seed"]}; identical raw features, train/validation/test row indices, train-fitted evaluation encoder and random-forest evaluator. Both GANs trained for {q["epochs"]} epochs. This is equal epochs, not equal compute or identical GAN internals. CTGAN uses its own internal preprocessing.','',
           '| Model | Test accuracy | Macro F1 | AUROC | Average precision |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:lines.append('| '+row['model']+' | '+' | '.join(f'{row[m]:.4f}' for m in ['accuracy','f1_macro','roc_auc','average_precision'])+' |')
    lines+=['','The two final-epoch rows are the direct duration-matched comparison. QTabGAN’s selected checkpoint is additional context, chosen by validation macro F1; CTGAN was evaluated only at its final epoch. This single-seed comparison must not be presented as a multi-seed model ranking. SDV CTGAN differs from the paper’s CTAB-GAN and CTAB-GAN+ baselines.','',
            '[ROC and precision–recall comparison](auc_comparison.png) · [Numbers and confusion counts](summary.csv)','',
            f'Source QTabGAN folder: `{a.qtab_run}`. Source CTGAN folder: `{a.ctgan_run}`. Each has loss curves and predictive diagnostics. CTGAN additionally retains raw numeric distribution and class-specific diversity diagnostics. Its model is the SDV implementation imported from the installed ctgan package, not a second implementation in qtabgan/.']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
