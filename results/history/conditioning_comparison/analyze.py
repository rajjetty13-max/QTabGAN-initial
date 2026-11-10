"""Rebuild aggregate artifacts from the three completed paired comparisons."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent
root=out.parent
frame=pd.concat([pd.read_csv(root/f'conditioning_comparison_seed_{seed}'/'summary.csv').assign(seed=seed) for seed in [42,43,44]],ignore_index=True)
frame.to_csv(out/'all_results.csv',index=False)
metrics=['accuracy','f1_macro','roc_auc','average_precision','mean_raw_numeric_ks','mean_minimum_within_class_std_ratio','mean_category_coverage']
aggregate=frame.groupby(['checkpoint','ansatz'])[metrics].agg(['mean','std'])
aggregate.to_csv(out/'aggregate.csv')
fig,axes=plt.subplots(2,3,figsize=(13,8),squeeze=False)
variants=['paper + D(x)','paper + D(x,y)']
for row,checkpoint in enumerate(['best','last']):
 for ax,metric in zip(axes[row],['f1_macro','roc_auc','average_precision']):
  for seed in [42,43,44]:
   pair=frame[(frame.seed==seed)&(frame.checkpoint==checkpoint)].set_index('ansatz')
   ax.plot([0,1],pair.loc[variants,metric],marker='o',label=f'Seed {seed}')
  ax.set(xticks=[0,1],xticklabels=['D(x)','D(x,y)'],ylim=(0,1),title=f'{checkpoint}: {metric}');ax.legend()
fig.suptitle('Paired test scores: validation-selected (best) and epoch 30 (last)')
fig.tight_layout();fig.savefig(out/'paired_scores.png');plt.close(fig)
lines=['# Paper-based discriminator conditioning: three-seed test','',
       'The original quantum circuit and mapper were retained. Only discriminator label conditioning was varied, following D(x,y) in paper §3.2. All six runs completed 30 epochs on full Loan data with paired splits and training seeds.','',
       '## Results','',
       'Means ± sample standard deviations over seeds 42, 43 and 44. “best” uses highest validation macro F1 from epochs 1/10/20/30, then reports performance on the shared per-seed test set. “last” reports epoch 30. These are exploratory results across three varying splits and initializations, not confidence intervals.','',
       '| Checkpoint | Discriminator | Accuracy | Macro F1 | AUROC | Average precision |',
       '| --- | --- | ---: | ---: | ---: | ---: |']
for checkpoint in ['best','last']:
 for variant in variants:
  part=frame[(frame.checkpoint==checkpoint)&(frame.ansatz==variant)]
  cells=[f'{part[m].mean():.4f} ± {part[m].std():.4f}' for m in metrics[:4]]
  lines.append('| '+ ' | '.join([checkpoint,variant,*cells])+' |')
lines+=['','## What this establishes','',
        'Validation-selected D(x,y) checkpoints improved macro F1 and AUROC on each of the three paired seeds. Their selected epoch was 10 for all seeds; controls selected epochs 1, 1 and 30. At the common final epoch 30, the conditional variant did not show a consistent utility improvement. This is evidence for an early-checkpoint benefit, not stable training or successful paper reproduction.','',
        'Both variants still collapse. Mean category coverage remains around 39–42%, raw numeric KS errors are large, and the mean minimum within-class standard-deviation ratio remains below 1% in the selected models. That summary averages the smaller class-specific ratio for each numeric feature; it does not assert every class-feature ratio is below 1%.','',
        'In seed 42, the mapper diagnostics at epoch 30 show quantum probabilities respond to noise, while fixed-label encoded outputs vary only weakly. This is consistent with loss of diversity in the generator/mapper path, not proof of a single causal bug. The diagnostic uses mixed encoded coordinates, so sensitivity magnitudes across spaces cannot be interpreted as a direct information-retention percentage.','',
        'The paper specifies ReLU hidden layers and a linear mapper output; those were preserved. No output sigmoid, auxiliary classifier or replacement loss was added. The next controlled paper-based experiment should investigate the unspecified optimization settings and mapper widths, using validation metrics and diversity together. Previously viewed test sets must not be used to select the next settings.','',
        '## Evidence','',
        '- [Paired score plots](paired_scores.png)',
        '- [All checkpoint numbers](all_results.csv)',
        '- [Aggregate numbers](aggregate.csv)']
for seed in [42,43,44]:
 lines.append(f'- [Seed {seed}: curves and complete metrics](../conditioning_comparison_seed_{seed}/REPORT.md)')
lines+=['','Each source run retains losses, confusion matrices, ROC/precision–recall curves and their underlying numbers, generated tables, raw distribution diagnostics, manifests, and best/latest checkpoints.','',
        'Seed 42 mapper diagnostics: [control](../conditioning_trial_30_control/mapper_last_diagnostic.json), [D(x,y)](../conditioning_trial_30_conditional/mapper_last_diagnostic.json).','',
        'Architecture assumption: the paper names D(x,y) but its diagram does not explicitly wire the class input. This experiment concatenates the one-hot class at the discriminator input. Historical feature-only checkpoints remain compatible via an opt-in flag. Analytic simulation/backprop, implementation-chosen hidden widths and dataset/evaluator assumptions remain deviations or unspecified details. No hardware or gate-noise simulation was used. All 25 unit tests passed.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(aggregate.to_string())
