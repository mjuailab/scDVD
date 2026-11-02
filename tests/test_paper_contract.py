from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from scdvd import scDVDModule
from scdvd.checkpoints import load_diffusion_checkpoint, validate_dataset_metadata
from scdvd.checkpoints import is_paper_model
from types import SimpleNamespace
from scdvd.data.datamodule import DiffusionDataModule
from scdvd.data.unpaired import DatasetMetadata
from test_regroup_padding import FakeStore


def config(name='sciplex3'):
    import scdvd
    directory = Path(scdvd.__file__).resolve().parent/'configs'
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        return compose(config_name=name)


class PaperContractTest(unittest.TestCase):
    def test_historical_checkpoints_are_not_paper_models(self):
        cfg = OmegaConf.to_container(config().model, resolve=True)
        model = SimpleNamespace(hparams={"model_config": cfg})
        self.assertTrue(is_paper_model(model))
        cfg['population']['set_size'] = 16
        self.assertFalse(is_paper_model(model))
        cfg['population']['set_size'] = 4
        cfg['denoiser']['population_gate_enabled'] = True
        self.assertFalse(is_paper_model(model))

    def test_defaults_and_portable_paths(self):
        for name in ['sciplex3', 'op3']:
            cfg = config(name)
            self.assertEqual(cfg.model.population.set_size, 4)
            self.assertFalse(cfg.model.denoiser.population_gate_enabled)
            self.assertIsNone(cfg.model.denoiser.population_correction_rms_ratio)
            self.assertEqual(cfg.model.objective.distribution_loss, 'geomloss_energy')
            self.assertEqual(cfg.model.objective.mmd_weight, .001)
            self.assertTrue(cfg.model.objective.mask_padded_cells)
            self.assertEqual(cfg.training.partition_set_size, 16)
            self.assertEqual(cfg.training.scheduler, 'cosine')
            cfg.paths.data_root = '/tmp/paper-data'
            cfg.repeat = 3; cfg.split = 5
            self.assertEqual(cfg.dataset.path, f'/tmp/paper-data/{name}/r3/s5/data.h5ad')

    def test_set4_preserves_reference_cell_membership(self):
        def make(size, reference):
            dm = DiffusionDataModule(dataset={}, population_set_size=size,
                partition_set_size=reference, sampling_strategy='sets',
                validation_fraction=.1, validation_seed=42, regroup_each_epoch=True)
            dm.store = FakeStore(); dm.setup('fit')
            return dm
        ref, final = make(16, None), make(4, 16)
        for attr in ['train_dataset', 'valid_dataset']:
            a, b = getattr(ref, attr), getattr(final, attr)
            np.testing.assert_array_equal(np.unique(np.concatenate(a.sets)), np.unique(np.concatenate(b.sets)))
            self.assertTrue(all(len(s)==4 for s in b.sets))
        for _ in range(3):
            final.train_dataloader()
            self.assertEqual(np.intersect1d(np.concatenate(final.train_dataset.sets),
                                           np.concatenate(final.valid_dataset.sets)).size, 0)

    def test_checkpoint_relocation_and_strict_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root/'features.npy'
            np.save(features, np.random.default_rng(42).normal(size=(3,8)).astype('float32'))
            cfg = config()
            cfg.model.chemical_feature_path = str(features)
            cfg.model.condition.output_dim = 8
            cfg.model.denoiser.hidden_dim = 16
            cfg.model.denoiser.num_heads = 2
            cfg.model.denoiser.cell_layers = cfg.model.denoiser.population_layers = 1
            cfg.model.diffusion.num_timesteps = 8
            metadata = dict(num_genes=6, num_cell_types=1, num_batches=1, num_local_drugs=3,
                            gene_names=tuple('abcdef'), gene_ids=tuple('abcdef'))
            model = scDVDModule(model_config=OmegaConf.to_container(cfg.model, resolve=True),
                                training_config=OmegaConf.to_container(cfg.training, resolve=True), metadata=metadata)
            self.assertFalse(any('population_gate.' in k for k in model.state_dict()))
            payload = {'hyper_parameters': deepcopy(dict(model.hparams)), 'state_dict': model.state_dict()}
            payload['hyper_parameters']['model_config']['chemical_feature_path'] = '/missing/original.npy'
            path = root/'test.ckpt';torch.save(payload,path)
            with self.assertRaises(FileNotFoundError):load_diffusion_checkpoint(path)
            loaded = load_diffusion_checkpoint(path, chemical_feature_path=features)
            for key,value in model.state_dict().items():torch.testing.assert_close(value,loaded.state_dict()[key],rtol=0,atol=0)
            bad = DatasetMetadata(6,1,1,3,('demo',),('b',),('a','b','c'),tuple('fedcba'),tuple('fedcba'))
            with self.assertRaisesRegex(ValueError,'ordering'):validate_dataset_metadata(loaded,bad)
            payload['state_dict'] = dict(payload['state_dict'])
            payload['state_dict'].pop(next(iter(payload['state_dict'])))
            torch.save(payload,path)
            with self.assertRaises(RuntimeError):load_diffusion_checkpoint(path, chemical_feature_path=features)


class AggregationTest(unittest.TestCase):
    def test_repeat_level_sd_and_missing_runs(self):
        path = Path(__file__).resolve().parents[1]/'scripts/aggregate_results.py'
        spec = importlib.util.spec_from_file_location('aggregate_results',path)
        mod = importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        rows = [dict(dataset='demo',evaluation_split='ood',genes='all',metric='rmse',repeat=r,split=s,value=r)
                for r in range(1,4) for s in range(1,6)]
        result = mod.aggregate(rows)[0]
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result['std'],1.)
        with self.assertRaises(ValueError):mod.aggregate(rows[:-1])
        with self.assertRaises(ValueError):mod.aggregate(rows+[rows[0]])


if __name__ == '__main__':unittest.main()
