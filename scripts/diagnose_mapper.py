"""Measure quantum-to-mapper variation and label effects without retraining."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from qtabgan import QTabGAN, QTabGANConfig


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=512)
    args=parser.parse_args()
    if args.samples<2: parser.error('Require at least two samples')
    if args.output.exists(): parser.error('Choose a fresh output file')
    torch.set_num_threads(1)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=QTabGANConfig(**state['config']);cfg.device='cpu';cfg.diff_method='backprop';cfg.finite_shot_training=False
    model=QTabGAN(state['data_dim'],state['condition_dim'],cfg)
    model.load_state_dict(state['state_dict']);model.eval()
    generator=torch.Generator().manual_seed(314159)
    noise=torch.rand(args.samples,cfg.n_qubits,generator=generator)*np.pi
    rows=[]
    with torch.no_grad():
        probabilities=model.quantum_generator(noise)
        outputs=[]
        for label in range(model.condition_dim):
            condition=torch.zeros(args.samples,model.condition_dim);condition[:,label]=1
            output=model.mapper(probabilities,condition)
            outputs.append(output)
            rows.append({'class_index':label,
                         'encoded_output_rms_std_over_noise':float(output.std(0,unbiased=False).square().mean().sqrt()),
                         'encoded_output_mean_abs':float(output.abs().mean())})
        pair_changes=[float((outputs[a]-outputs[b]).square().mean().sqrt())
                      for a in range(len(outputs)) for b in range(a+1,len(outputs))]
        delta=torch.randn(noise.shape,generator=generator)*.001
        moved_p=model.quantum_generator(noise+delta)
        c=torch.zeros(args.samples,model.condition_dim);c[:,0]=1
        moved=model.mapper(moved_p,c)
        metrics={'checkpoint':str(args.checkpoint),'epochs_trained':state['epochs_trained'],
                 'seed':314159,'latent_samples':args.samples,'analytic_diagnostic':True,
                 'quantum_probability_rms_std_over_noise':float(probabilities.std(0,unbiased=False).square().mean().sqrt()),
                 'quantum_probability_mean_l2_norm':float(probabilities.norm(dim=1).mean()),
                 'class_indicator_l2_norm':1.0,'per_class':rows,
                 'mean_class_switch_rms_output_change_at_same_noise':float(np.mean(pair_changes)) if pair_changes else None,
                 'angle_perturbation_rms':float(delta.square().mean().sqrt()),
                 'probability_perturbation_rms':float((moved_p-probabilities).square().mean().sqrt()),
                 'encoded_output_perturbation_rms_class_0':float((moved-outputs[0]).square().mean().sqrt()),
                 'interpretation':'Encoded coordinates mix numeric and one-hot features. These sensitivity summaries locate attenuation, not a proven cause of collapse.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
