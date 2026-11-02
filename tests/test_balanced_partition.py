import unittest
from types import SimpleNamespace
import numpy as np
from scdvd.data.datamodule import DiffusionDataModule
from scdvd.data.unpaired import PopulationPerturbationDataset

class FakeStore:
    def population_dataset(self, split, set_size, max_samples, seed):
        ds = PopulationPerturbationDataset.__new__(PopulationPerturbationDataset)
        ds.set_size = set_size
        ds._regroup_pool = {(0,0,0,0,1.,24.,1,0): np.arange(1600),
                           (1,0,1,1,1.,24.,1,0): np.arange(1600,3200)}
        ds.sets = ds._make_sets(seed)
        return ds

class PartitionTest(unittest.TestCase):
    def test_sizes_keep_reference_pool_validation_and_exposure(self):
        reference = None
        for size in (16,32,64,100):
            dm = DiffusionDataModule(dataset={}, population_set_size=size,
                partition_set_size=16, sampling_strategy='cell_drug_balanced',
                validation_fraction=.1, max_eval_samples=2048)
            dm.store = FakeStore()
            dm.setup('fit')
            valid = np.stack(dm.valid_dataset.sets)
            pool = np.sort(np.concatenate(list(dm.train_dataset._regroup_pool.values())))
            if reference is None:
                reference = (valid,pool,len(dm.train_dataset))
            np.testing.assert_array_equal(reference[0], valid)
            np.testing.assert_array_equal(reference[1], pool)
            self.assertEqual(len(dm.train_dataset), reference[2])
            self.assertFalse(np.intersect1d(pool,valid).size)
            dm.train_dataloader()
            self.assertEqual(np.stack(dm.train_dataset.sets).shape, (reference[2],size))
            self.assertEqual(dm.valid_dataset.set_size,16)

if __name__ == '__main__':
    unittest.main()
