"""Checks for hierarchical population sampling, independent of expression data."""
import unittest
from collections import Counter
import numpy as np
from scdvd.data.unpaired import PopulationPerturbationDataset


class BalancedSamplingTest(unittest.TestCase):
    def test_balance_homogeneity_partition_and_reproducibility(self):
        ds = PopulationPerturbationDataset.__new__(PopulationPerturbationDataset)
        ds.set_size = 16
        pools = {}
        cursor = 0
        for cell, drug, dose, batch, n in [(0,0,1.,0,3),(0,0,1.,1,30),
                                         (0,0,2.,0,300),(1,1,1.,0,3000)]:
            key = (cell,batch,drug,drug,dose,24.,int(dose),0)
            pools[key] = np.arange(cursor,cursor+n)
            cursor += n
        ds._regroup_pool = pools
        retained = [v[:-1] for v in pools.values()]
        held_out = {int(v[-1]) for v in pools.values()}
        ds.restrict_regroup_pool_to_sets(retained)
        ds.sets = [np.zeros(16,dtype=int)] * 20000
        ds.resample_cell_drug_balanced(42)
        first = np.stack(ds.sets)
        lookup = {int(i): k for k,v in ds._regroup_pool.items() for i in v}
        pairs, doses, batches = Counter(), Counter(), Counter()
        for indices in ds.sets:
            keys = {lookup[int(i)] for i in indices}
            self.assertEqual(len(keys),1)
            self.assertFalse(held_out.intersection(indices))
            key = keys.pop()
            self.assertEqual(len(np.unique(indices)), min(16,len(ds._regroup_pool[key])))
            pairs[key[0],key[2]] += 1
            if key[0] == 0:
                doses[key[4]] += 1
                if key[4] == 1.:
                    batches[key[1]] += 1
        self.assertLess(abs(pairs[0,0]/20000 - .5), .02)
        self.assertLess(abs(doses[1.]/sum(doses.values()) - .5), .025)
        self.assertLess(abs(batches[0]/sum(batches.values()) - 2/31), .025)
        ds.resample_cell_drug_balanced(42)
        np.testing.assert_array_equal(first,np.stack(ds.sets))
        ds.resample_cell_drug_balanced(43)
        self.assertEqual(len(ds.sets),20000)
        self.assertFalse(np.array_equal(first,np.stack(ds.sets)))

if __name__ == '__main__':
    unittest.main()
