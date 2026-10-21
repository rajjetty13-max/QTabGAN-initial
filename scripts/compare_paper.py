"""Compare completed QTabGAN runs with Table 3; never certify a reproduction."""
import argparse
import json
from pathlib import Path
import pandas as pd

REFERENCE = {
    'adult': {'real_accuracy': .7760, 'synthetic_accuracy': .7740, 'accuracy_gap': .0020,
              'real_f1': .7619, 'synthetic_f1': .7539, 'f1_gap': .0080},
    'loan': {'real_accuracy': .9600, 'synthetic_accuracy': .9380, 'accuracy_gap': .0220,
             'real_f1': .9570, 'synthetic_f1': .9291, 'f1_gap': .0279},
}
p = argparse.ArgumentParser(__doc__)
p.add_argument('runs', type=Path, nargs='+')
p.add_argument('--output', type=Path, default=Path('results/paper_comparison.csv'))
a = p.parse_args(); rows = []
for root in a.runs:
    manifest = json.loads((root / 'manifest.json').read_text())
    name = manifest['arguments']['dataset_name']
    reference = REFERENCE.get(name)
    if reference is None: continue
    for metrics_file in sorted(root.glob('seed_*/*/metrics.json')):
        scores = json.loads(metrics_file.read_text())
        if scores.get('model') != 'qtabgan':
            continue  # Table 3 reports QTabGAN, not CTGAN.
        real = json.loads((metrics_file.parents[1] / 'real/utility.json').read_text())
        row = {'dataset': name, 'model': scores['model'], 'seed': scores['seed'],
               'epochs': scores['epochs'], 'real_accuracy': real['accuracy'],
               'synthetic_accuracy': scores['accuracy'], 'accuracy_gap': scores['accuracy_gap'],
               'paper_real_accuracy':reference['real_accuracy'],
               'paper_synthetic_accuracy':reference['synthetic_accuracy'],
               'paper_accuracy_gap':reference['accuracy_gap'],
               'accuracy_gap_minus_paper':scores['accuracy_gap']-reference['accuracy_gap'],
               'paper_real_f1':reference['real_f1'], 'paper_synthetic_f1':reference['synthetic_f1'],
               'paper_f1_gap':reference['f1_gap'], 'paper_accuracy_gap_met': round(scores['accuracy_gap'], 4) <= reference['accuracy_gap'],
               'reproduction_status': ('gap_only_protocol_unverified' if round(scores['accuracy_gap'], 4) <= reference['accuracy_gap']
                                       else 'reported_accuracy_gap_not_matched')}
        for convention in ['macro','weighted','binary']:
            key = 'f1_'+convention
            if key in scores:
                row['synthetic_'+key] = scores[key]
                row['real_'+key] = real[key]
                row[key+'_gap'] = abs(real[key]-scores[key])
        rows.append(row)
a.output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.output, index=False)
print(pd.DataFrame(rows).to_string(index=False))
