"""Untrained circuit diagnostics; does not claim improved GAN utility or use hardware."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import torch
import pennylane as qml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qtabgan.quantum_generator import QuantumGeneratorCore
from qtabgan.ionq import export_qis, simulate_qis


def diagnose(ansatz, seed, samples, out):
    torch.manual_seed(seed)
    core = QuantumGeneratorCore(8, 2, ansatz=ansatz).double()
    generator = torch.Generator().manual_seed(seed + 1000)
    noise = torch.rand(samples, 8, generator=generator, dtype=torch.float64)*np.pi
    weights = core.weights.detach()
    with torch.no_grad():
        p = core.circuit(noise, weights).numpy()
        derivatives = []
        for i in range(weights.numel()):
            shift = torch.zeros_like(weights); shift.reshape(-1)[i] = np.pi/2
            derivative = (core.circuit(noise, weights+shift)-core.circuit(noise, weights-shift))/2
            derivatives.append(derivative.numpy())
    jacobian = np.stack(derivatives, axis=-1).reshape(-1, weights.numel())
    norms = np.sqrt(np.mean(jacobian**2, axis=0))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tape = qml.tape.make_qscript(core._probabilities)(noise[0], weights)
    resources = qml.specs(core.circuit)(noise[0], weights)["resources"]
    qis = export_qis(core, noise[0])
    export_error = float(np.max(np.abs(simulate_qis(qis)-p[0])))
    row = {'ansatz': ansatz, 'seed': seed, 'qubits': 8, 'layers': 2,
           'gates': resources.num_gates, 'depth': resources.depth,
           'cnot_count': sum(op.name=='CNOT' for op in tape.operations),
           'parameters': weights.numel(), 'active_parameters': int((norms>1e-9).sum()),
           'stacked_probability_jacobian_rank': int((singular>singular[0]*1e-8).sum()),
           'mean_pair_total_variation': float(np.abs(p[:,None,:]-p[None,:,:]).sum(-1)[np.triu_indices(samples,1)].mean()/2),
           'mean_normalized_entropy': float(-(p*np.log(np.maximum(p,1e-300))).sum(-1).mean()/np.log(p.shape[-1])),
           'ionq_export_max_probability_error': export_error}
    directory = out/f'{ansatz}_seed_{seed}'; directory.mkdir()
    pd.DataFrame({'parameter': np.arange(len(norms)), 'probability_derivative_rms': norms}).to_csv(directory/'parameter_sensitivity.csv',index=False)
    pd.DataFrame({'index': np.arange(len(singular)), 'singular_value': singular}).to_csv(directory/'jacobian_singular_values.csv',index=False)
    (directory/'circuit.txt').write_text(qml.draw(core.circuit)(noise[0],weights))
    fig,_ = qml.draw_mpl(core.circuit, decimals=None)(noise[0],weights)
    fig.savefig(directory/'circuit.png',bbox_inches='tight'); plt.close(fig)
    (directory/'metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/circuit_comparison'))
    parser.add_argument('--ansatze', nargs='+', choices=['paper','reuploading','paper_readout'], default=['paper','reuploading'])
    parser.add_argument('--samples',type=int,default=32)
    args = parser.parse_args()
    if args.samples<2: parser.error('At least two latent samples required')
    if args.output.exists(): parser.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    manifest = {'seeds': [42,43,44], 'ansatze': args.ansatze, 'latent_samples_per_seed': args.samples,
                'training_performed': False, 'hardware_used': False,
                'active_derivative_rms_threshold': 1e-9, 'relative_rank_tolerance': 1e-8,
                'depth_definition': 'Logical gate dependency depth, not backend native/transpiled depth',
                'versions': {name: importlib.metadata.version(name) for name in ['torch','pennylane','numpy']},
                'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  [Path(__file__), Path('qtabgan/quantum_generator.py')]}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    for seed in (42,43,44):
        for ansatz in args.ansatze:
            row=diagnose(ansatz,seed,args.samples,args.output);rows.append(row)
            print(json.dumps(row),flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(args.output/'summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,['depth','active_parameters','mean_pair_total_variation']):
        frame.groupby('ansatz')[metric].mean().plot.bar(ax=ax,rot=0)
        ax.set_title(metric);ax.set_xlabel('')
    fig.tight_layout();fig.savefig(args.output/'comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for ansatz in args.ansatze:
        data=pd.read_csv(args.output/f'{ansatz}_seed_42'/'parameter_sensitivity.csv')
        ax.plot(data.parameter,data.probability_derivative_rms,marker='.',label=ansatz)
    ax.set(xlabel='Parameter index',ylabel='RMS probability derivative',title='Seed 42: sensitivity of probabilities to each angle')
    ax.legend();fig.tight_layout();fig.savefig(args.output/'parameter_sensitivity.png');plt.close(fig)


if __name__=='__main__': main()
