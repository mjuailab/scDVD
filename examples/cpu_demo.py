"""Small synthetic CPU exercise of training, checkpoint loading and masked evaluation.

This is an installation example, not a biological benchmark or paper result.
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
        'dose': np.asarray([0.]*12+[1.]*32+[2.]*12+[1.]*12),
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
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    make_data(directory)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    common = [
        f'dataset.path={directory}/demo.h5ad', f'dataset.pert_mapping_path={directory}/drug_lookup.csv',
        'dataset.dose_divisor=1', f'model.chemical_feature_path={directory}/unimol.npy',
        f'paths.output_root={directory}/outputs', 'model.condition.output_dim=8',
        'model.denoiser.hidden_dim=16', 'model.denoiser.cell_layers=1',
        'model.denoiser.population_layers=1', 'model.denoiser.num_heads=2',
        'model.denoiser.mlp_ratio=2', 'model.mask.hidden_dim=16',
        'model.mask.condition_dim=8', 'model.mask.num_blocks=1',
        'model.diffusion.num_timesteps=8', 'training.accelerator=cpu',
        'training.num_workers=0', 'training.pin_memory=false', 'training.max_epochs=1',
        'training.batch_size=2', 'training.accumulate_grad_batches=1',
        'training.run_id=diffusion', 'mask_training.accelerator=cpu',
        'mask_training.num_workers=0', 'mask_training.pin_memory=false',
        'mask_training.max_epochs=1', 'mask_training.batch_size=8', 'mask_training.run_id=mask',
    ]
    def run(command, extra=()):
        subprocess.run([sys.executable, '-m', 'scdvd', command, *common, *extra], env=env, check=True)
    run('train')
    run('train-mask')
    def best(name):
        files = [p for p in (directory/'outputs'/name).glob('*.ckpt') if p.name != 'last.ckpt']
        if len(files) != 1:
            raise RuntimeError(f'Expected one best checkpoint for {name}: {files}')
        return files[0]
    run('evaluate', [f'checkpoint={best("diffusion")}', f'mask_checkpoint={best("mask")}',
                     'evaluation.device=cpu', 'evaluation.ddim_steps=4',
                     'evaluation.save_predictions=true', f'evaluation.output_dir={directory}/evaluation'])
    with (directory/'evaluation/condition_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and int(rows[0]['generated_cells']) == 12
    prediction = np.load(directory/'evaluation/predictions/condition_0001.npz')
    assert prediction['scDVD'].shape == prediction['GT'].shape == (12,48)
    assert np.isfinite(prediction['scDVD']).all()
    print(f'CPU demo passed. Results: {directory}/evaluation')


if __name__ == '__main__':
    main()
