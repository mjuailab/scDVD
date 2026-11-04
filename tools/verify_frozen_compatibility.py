"""CPU regression against the exact source and checkpoint of a completed run."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from scdvd.checkpoints import load_diffusion_checkpoint
from scdvd.runtime import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Frozen source root containing scdvd/')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    package = args.reference/'scdvd'
    spec = importlib.util.spec_from_file_location('_frozen_scdvd',package/'__init__.py',submodule_search_locations=[str(package)])
    old = importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    reference = old.scDVDModule.load_from_checkpoint(args.checkpoint,map_location='cpu').eval()
    current = load_diffusion_checkpoint(args.checkpoint,map_location='cpu').eval()
    if reference.state_dict().keys() != current.state_dict().keys():
        raise AssertionError('Checkpoint parameter names changed')
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value,current.state_dict()[key],rtol=0,atol=0)
    torch.manual_seed(117)
    genes = int(current.hparams['metadata']['num_genes'])
    source = torch.rand(5,genes)*.1
    batch = {'source':source,'control_mean':source.mean(0,keepdim=True).expand_as(source),
             'cell_idx':torch.zeros(5,dtype=torch.long),'batch_idx':torch.tensor([0,0,0,0,1]),
             'pert_idx':torch.zeros(5,dtype=torch.long),'local_drug_idx':torch.zeros(5,dtype=torch.long),
             'dosage':torch.ones(5)}
    torch.manual_seed(42);expected=reference.bridge_sample(batch,ddim_steps=4)
    torch.manual_seed(42);actual=current.bridge_sample(batch,ddim_steps=4)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Initial parameter values and random-number consumption must also match.
    parameters = dict(current.hparams)
    torch.manual_seed(97); fresh_old = old.scDVDModule(**parameters)
    torch.manual_seed(97); fresh_new = type(current)(**parameters)
    for key, value in fresh_old.state_dict().items():
        torch.testing.assert_close(value,fresh_new.state_dict()[key],rtol=0,atol=0)
    result={'passed':True,'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha256(args.checkpoint),
            'parameter_keys_identical':True,'initialization_exact':True,
            'cpu_ddim_output_exact':True,'ddim_steps':4,'generated_cells':5,
            'max_absolute_difference':float((actual-expected).abs().max())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
