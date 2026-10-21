"""Plot holdout utility of completed runs alongside the real-trained reference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(__doc__)
p.add_argument('runs', type=Path, nargs='+')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--output', type=Path, default=Path('results/comparison.png'))
a = p.parse_args()
fig, axes = plt.subplots(len(a.runs), 3, figsize=(12, 3.5*len(a.runs)), squeeze=False)
metrics = [('accuracy','Accuracy'),('f1_macro','Macro F1'),('roc_auc','AUROC')]
for row, root in enumerate(a.runs):
    name = json.loads((root/'manifest.json').read_text())['arguments']['dataset_name']
    base = root/f'seed_{a.seed}'
    results = [json.loads((base/'real/utility.json').read_text())]
    for model in ['ctgan','qtabgan']:
        results.append(json.loads((base/model/'metrics.json').read_text()))
    for col, (metric, title) in enumerate(metrics):
        ax=axes[row,col]
        values=[r.get(metric,0.) for r in results]
        bars=ax.bar(['Real-trained','CTGAN','QTabGAN'],values,color=['#777777','#0072B2','#D55E00'])
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set(ylim=(0,1.08),title=f'{name}: {title}')
        ax.spines[['top','right']].set_visible(False)
fig.suptitle(f'Train on real or synthetic; test on the same held-out real rows (seed {a.seed})')
fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=160);plt.close(fig)
