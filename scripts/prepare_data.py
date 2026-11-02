"""Apply an exact released cell assignment to an already preprocessed H5AD.

Does not select HVGs, normalize raw counts, or generate new benchmark splits.
Requires the optional data dependencies: pip install '.[data]'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd


def apply_assignment(adata, assignments):
    cell_column = assignments.columns[0]
    ids = assignments[cell_column].astype(str)
    if ids.duplicated().any() or not adata.obs_names.is_unique:
        raise ValueError("Cell identifiers must be unique")
    if set(ids) != set(adata.obs_names.astype(str)):
        raise ValueError("Input cells do not exactly match the released split assignment")
    aligned = assignments.set_index(cell_column).loc[adata.obs_names.astype(str)]
    for column in ("condition", "cell_type"):
        if not np.array_equal(adata.obs[column].astype(str), aligned[column].astype(str)):
            raise ValueError(f"Assignment metadata disagrees with H5AD: {column}")
    if not np.allclose(adata.obs['dose_val'].astype(float), aligned['dose_val'].astype(float), rtol=0, atol=1e-7):
        raise ValueError("Assignment doses disagree with preprocessed H5AD")
    control = adata.obs['control'].astype(str).str.lower().isin(['true','1'])
    recorded_control = aligned['control'].astype(str).str.lower().isin(['true','1'])
    if not np.array_equal(control, recorded_control):
        raise ValueError("Assignment control labels disagree with H5AD")
    values = aligned['unlasting_split'].astype(str)
    if not set(values).issubset({'train','test','ood'}):
        raise ValueError("Unknown split labels")
    if not (values.to_numpy()[control.to_numpy()] == 'train').all():
        raise ValueError("All controls must remain in training")
    adata.obs['unlasting_split'] = pd.Categorical(values.to_numpy())
    return adata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['sciplex3','op3'], required=True)
    parser.add_argument('--h5ad', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--lookup', type=Path, required=True)
    parser.add_argument('--repeat', type=int, choices=range(1,4), default=1)
    parser.add_argument('--split', type=int, choices=range(1,6), default=1)
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    released = project/'paper/splits'/args.dataset/f'r{args.repeat}'/f's{args.split}'
    output = args.output/args.dataset/f'r{args.repeat}'/f's{args.split}'/'data.h5ad'
    if output.exists():
        raise FileExistsError(output)
    features_dir = args.output/args.dataset/'features'
    # Verify before creating the output. The released feature row order is fixed.
    from scdvd.runtime import sha256
    expected = json.loads((project/'paper/manifests/assets.json').read_text())[args.dataset]
    for source, name in [(args.features,'unimol.npy'),(args.lookup,'drug_lookup.csv')]:
        if sha256(source) != expected[name]['sha256']:
            raise ValueError(f'Wrong paper asset: {name}')
        target = features_dir/name
        if target.exists() and sha256(target) != expected[name]['sha256']:
            raise FileExistsError(f'Different existing feature asset: {target}')
    adata = ad.read_h5ad(args.h5ad)
    if adata.n_vars != 2000:
        raise ValueError('The paper uses the fixed 2,000-HVG expression space')
    genes = json.loads((project/'paper/manifests'/f'{args.dataset}_genes.json').read_text())
    if list(adata.var_names.astype(str)) != genes:
        raise ValueError('Gene identity/order differs from the paper')
    assignments = pd.read_csv(released/'cell_assignments.csv.gz')
    apply_assignment(adata, assignments)
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output, compression='gzip')
    features_dir.mkdir(parents=True, exist_ok=True)
    for source, name in [(args.features,'unimol.npy'),(args.lookup,'drug_lookup.csv')]:
        if not (features_dir/name).exists():
            shutil.copy2(source, features_dir/name)
    print(output)


if __name__ == '__main__':
    main()
