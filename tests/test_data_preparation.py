import importlib.util
from pathlib import Path
import unittest

import numpy as np

try:
    import anndata as ad
    import pandas as pd
except ImportError:
    ad = None


@unittest.skipIf(ad is None, 'Install scdvd[data] to test prepared H5AD assignments')
class PreparedDataTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]/'scripts/prepare_data.py'
        spec = importlib.util.spec_from_file_location('prepare_data',path)
        self.module = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        obs = pd.DataFrame({'condition':['control','drug','drug'], 'cell_type':['c']*3,
                            'dose_val':[0.,1.,2.], 'control':[True,False,False]},index=['c0','c1','c2'])
        self.adata = ad.AnnData(np.arange(6,dtype='float32').reshape(3,2),obs=obs)
        self.assignment = pd.DataFrame({'cell':['c2','c0','c1'], 'condition':['drug','control','drug'],
                                        'cell_type':['c']*3, 'dose_val':[2.,0.,1.],
                                        'control':[False,True,False], 'unlasting_split':['test','train','train']})

    def test_aligns_by_cell_identity_without_changing_expression(self):
        expected = self.adata.X.copy()
        self.module.apply_assignment(self.adata,self.assignment)
        self.assertEqual(list(self.adata.obs.unlasting_split),['train','train','test'])
        np.testing.assert_array_equal(expected,self.adata.X)

    def test_rejects_mismatched_cells_and_nontraining_controls(self):
        bad = self.assignment.copy();bad.loc[0,'cell']='missing'
        with self.assertRaisesRegex(ValueError,'exactly match'):
            self.module.apply_assignment(self.adata,bad)
        bad = self.assignment.copy();bad.loc[1,'unlasting_split']='ood'
        with self.assertRaisesRegex(ValueError,'controls'):
            self.module.apply_assignment(self.adata,bad)


if __name__=='__main__':unittest.main()
