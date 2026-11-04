"""Small synthetic exercise of training, checkpoint loading and masked evaluation.

Defaults to CPU. This is an installation example, not a biological benchmark.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch


def make_data(directory):
    rng = np.random.default_rng(42)
    n, genes = 68, 48
    values = rng.gamma(2, .1, (n, genes)).astype('float32')
    values[rng.random(values.shape) < .4] = 0
    obs = {
        'cell_id': np.asarray([f'cell_{i}' for i in range(n)], dtype=object),
        'cell_type': np.asarray(['demo']*n, dtype=object),
        'batch': np.asarray(['batch0']*n, dtype=object),
        'condition': np.asarray(['control']*12 + ['drug_a']*44 + ['drug_b']*12, dtype=object),
        'control': np.asarray([1]*12+[0]*56),
        'unlasting_split': np.asarray(['train']*44+['test']*12+['ood']*12, dtype=object),
        'dose': np.asarray([0.]*12+[1000.]*32+[2000.]*12+[1000.]*12),
        'dose_val': np.asarray([0.]*12+[1.]*32+[2.]*12+[1.]*12),
        'xpert_time': np.full(n, 24.),
        'xpert_time_idx': np.zeros(n, dtype='int64'),
        'xpert_dose_idx': np.asarray([0]*12+[1]*32+[2]*12+[1]*12),
        'xpert_pert_idx': np.asarray([0]*12+[1]*44+[2]*12),
    }
    with h5py.File(directory/'demo.h5ad', 'w') as f:
        f.create_dataset('X', data=values)
        group = f.create_group('obs'); group.attrs['_index'] = 'cell_id'
        for key, value in obs.items():
            group.create_dataset(key, data=value, dtype=h5py.string_dtype() if value.dtype.kind == 'O' else None)
        var = f.create_group('var');var.attrs['_index'] = 'gene'
        var.create_dataset('gene', data=np.asarray([f'gene_{i}' for i in range(genes)], dtype=object), dtype=h5py.string_dtype())
    np.save(directory/'unimol.npy', rng.normal(size=(3,8)).astype('float32'))
    (directory/'drug_lookup.csv').write_text('drug_name,drug_idx\ncontrol,0\ndrug_a,1\ndrug_b,2\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config-name', choices=['sciplex3', 'op3'], default='sciplex3')
    parser.add_argument('--evaluation-split', choices=['test', 'ood'], default='test')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; install a CUDA PyTorch build and check the NVIDIA driver')
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    make_data(directory)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    accelerator = 'gpu' if args.device == 'cuda' else 'cpu'
    common = [
        f'dataset.path={directory}/demo.h5ad', f'dataset.pert_mapping_path={directory}/drug_lookup.csv',
        f'model.chemical_feature_path={directory}/unimol.npy',
        f'paths.output_root={directory}/outputs', 'model.condition.output_dim=8',
        'model.denoiser.hidden_dim=16', 'model.denoiser.cell_layers=1',
        'model.denoiser.population_layers=1', 'model.denoiser.num_heads=2',
        'model.denoiser.mlp_ratio=2', 'model.mask.hidden_dim=16',
        'model.mask.condition_dim=8', 'model.mask.num_blocks=1',
        'model.diffusion.num_timesteps=8', f'training.accelerator={accelerator}',
        'training.num_workers=0', 'training.pin_memory=false', 'training.max_epochs=1',
        'training.batch_size=2', 'training.accumulate_grad_batches=1',
        'training.run_id=diffusion', f'mask_training.accelerator={accelerator}',
        'mask_training.num_workers=0', 'mask_training.pin_memory=false',
        'mask_training.max_epochs=1', 'mask_training.batch_size=8', 'mask_training.run_id=mask',
    ]
    def run(command, extra=()):
        subprocess.run([sys.executable, '-m', 'scdvd', command, '--config-name',
                        args.config_name, *common, *extra], env=env, check=True)
    run('train')
    run('train-mask')
    def best(name):
        files = [p for p in (directory/'outputs'/name).glob('*.ckpt') if p.name != 'last.ckpt']
        if len(files) != 1:
            raise RuntimeError(f'Expected one best checkpoint for {name}: {files}')
        checkpoint = torch.load(files[0], map_location='cpu', weights_only=False)
        assert checkpoint['global_step'] > 0, f'{name} did not take an optimizer step'
        assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
        # The best-validation score must be finite for both training stages.
        scores = [state['best_model_score'] for state in checkpoint['callbacks'].values()
                  if isinstance(state, dict) and 'best_model_score' in state]
        assert len(scores) == 1 and torch.isfinite(scores[0]), f'{name} validation failed'
        return files[0]
    run('evaluate', [f'checkpoint={best("diffusion")}', f'mask_checkpoint={best("mask")}',
                     f'evaluation.device={args.device}', 'evaluation.ddim_steps=4',
                     f'evaluation.split={args.evaluation_split}',
                     'evaluation.save_predictions=true', f'evaluation.output_dir={directory}/evaluation'])
    with (directory/'evaluation/condition_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and int(rows[0]['generated_cells']) == 12
    assert rows[0]['drug'] == ('drug_b' if args.evaluation_split == 'ood' else 'drug_a')
    for scope in ['all', 'de20', 'de40', 'bg']:
        for metric in ['rmse', 'e_distance', 'emd']:
            assert np.isfinite(float(rows[0][f'{scope}_{metric}']))
    prediction = np.load(directory/'evaluation/predictions/condition_0001.npz')
    assert prediction['scDVD'].shape == prediction['GT'].shape == (12,48)
    assert np.isfinite(prediction['scDVD']).all()
    assert (prediction['scDVD'] >= 0).all() and prediction['scDVD'].std() > 0
    print(f'{args.device.upper()} demo passed ({args.config_name}, {args.evaluation_split}). '
          f'Results: {directory}/evaluation')


if __name__ == '__main__':
    main()
