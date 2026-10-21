"""Inspect original-column fidelity; avoid one-hot averages hiding failures."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from qtabgan.evaluation import correlation_difference

p = argparse.ArgumentParser(__doc__)
p.add_argument('runs', type=Path, nargs='+')
a = p.parse_args()
for root in a.runs:
    manifest = json.loads((root / 'manifest.json').read_text())
    args = manifest['arguments']
    if args['data'] is None: continue
    original = pd.read_csv(args['data'])
    target = args['target']; cat = args['categorical']
    numerical = [c for c in original if c not in cat + [target]]
    for path in sorted(root.glob('seed_*/*/synthetic.csv')):
        split = np.load(path.parents[1] / 'split.npz')
        real = original.iloc[split['train']]; fake = pd.read_csv(path)
        rows = []; categorical = []
        for col in numerical:
            std = float(real[col].std())
            rows.append({'feature':col, 'ks':float(ks_2samp(real[col],fake[col],method='asymp').statistic),
                         'std_ratio':float(fake[col].std()/std) if std > 0 else None,
                         'real_mean':float(real[col].mean()), 'synthetic_mean':float(fake[col].mean()),
                         'synthetic_outside_train_1_99_fraction':float(((fake[col] < real[col].quantile(.01)) |
                                                                      (fake[col] > real[col].quantile(.99))).mean())})
        for col in cat:
            r, f = real[col].astype(str), fake[col].astype(str)
            categories = sorted(set(r) | set(f))
            rp = r.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
            fp = f.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
            categorical.append({'feature':col, 'jsd':float(jensenshannon(rp,fp)**2),
                                'coverage':len(set(r)&set(f))/len(set(r)),
                                'real_categories':int(r.nunique()), 'synthetic_categories':int(f.nunique())})
        conditional = []
        for label in real[target].unique():
            for col in numerical:
                r = real.loc[real[target] == label, col]
                f = fake.loc[fake[target] == label, col]
                std = float(r.std())
                ratio = float(f.std()/std) if std > 0 and len(f)>1 else None
                conditional.append({'class':str(label), 'feature':col, 'std_ratio':ratio})
        pd.DataFrame(rows).to_csv(path.parent / 'original_numerical_diagnostics.csv', index=False)
        pd.DataFrame(categorical).to_csv(path.parent / 'original_categorical_diagnostics.csv', index=False)
        pd.DataFrame(conditional).to_csv(path.parent / 'conditional_diversity.csv', index=False)
        summary = {'numerical_unique_pair_correlation_difference':correlation_difference(real[numerical].to_numpy(), fake[numerical].to_numpy()) if numerical else None,
                   'mean_original_numerical_ks':float(np.mean([r['ks'] for r in rows])) if rows else None,
                   'mean_original_categorical_jsd':float(np.mean([r['jsd'] for r in categorical])) if categorical else None,
                   'mean_category_coverage':float(np.mean([r['coverage'] for r in categorical])) if categorical else None,
                   'real_label_proportions':{str(k):float(v) for k,v in real[target].value_counts(normalize=True).items()},
                   'synthetic_label_proportions':{str(k):float(v) for k,v in fake[target].value_counts(normalize=True).items()},
                   'majority_class_test_accuracy':float((original.iloc[split['test']][target] == real[target].mode()[0]).mean())}
        (path.parent / 'original_column_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        fig, axes = plt.subplots(2, 3, figsize=(12,7))
        for ax, col in zip(axes.flat, numerical[:6]):
            for frame, label in [(real,'real train'),(fake,'synthetic')]:
                ax.plot(np.sort(frame[col]), np.arange(1,len(frame)+1)/len(frame),label=label)
            ax.set_title(col); ax.legend()
        for ax in list(axes.flat)[len(numerical[:6]):]: ax.set_visible(False)
        fig.tight_layout(); fig.savefig(path.parent / 'original_cdf.png'); plt.close(fig)
        print(path.parent, json.dumps(summary))
