"""Check regroup coverage, padding losses and exact inference counts."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scdvd.data.datamodule import DiffusionDataModule
from scdvd.data.unpaired import PopulationPerturbationDataset
from scdvd.module import raw_energy_distance, multiscale_rbf_mmd, support_weighted_expression_mse, scDVDModule

class FakeStore:
    def population_dataset(self, split, set_size, max_samples, seed):
        ds=PopulationPerturbationDataset.__new__(PopulationPerturbationDataset)
        ds.set_size=set_size
        ds._regroup_pool={(0,0,0,0,1.,24.,1,0):np.arange(600),(0,1,0,0,1.,24.,1,0):np.arange(600,617)}
        ds.sets=ds._make_sets(seed)
        return ds

class RegroupPaddingTest(unittest.TestCase):
    def test_coverage_new_combinations_and_validation_isolation(self):
        dm=DiffusionDataModule(dataset={},population_set_size=16,batch_size=8,
            regroup_each_epoch=True,drop_last_train=False,num_workers=0,
            validation_fraction=.1,validation_seed=1337)
        dm.store=FakeStore();dm.setup('fit')
        heldout=np.concatenate(dm.valid_dataset.sets).copy()
        pool=np.sort(np.concatenate(list(dm.train_dataset._regroup_pool.values())))
        previous=None
        for _ in range(3):
            loader=dm.train_dataloader()
            self.assertFalse(loader.drop_last)
            self.assertEqual(sum(len(x) for x in loader.batch_sampler),len(dm.train_dataset))
            sets=np.stack(dm.train_dataset.sets)
            np.testing.assert_array_equal(np.unique(sets),pool)
            self.assertFalse(np.intersect1d(sets,heldout).size)
            for indices in sets:
                self.assertEqual(len(set(indices<600)),1)
                self.assertEqual(len(indices),16)
            if previous is not None:self.assertFalse(np.array_equal(sets,previous))
            previous=sets.copy()
            np.testing.assert_array_equal(np.concatenate(dm.valid_dataset.sets),heldout)
        ds=FakeStore().population_dataset('train',16,None,42)
        self.assertEqual(len(ds),40)
        np.testing.assert_array_equal(np.unique(np.stack(ds.sets)),np.arange(617))

    def test_padded_rows_do_not_contribute_to_losses(self):
        x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]]);y=torch.tensor([[[2.,2.],[2.,5.],[6.,5.]]])
        xp=torch.cat([x,torch.full((1,13,2),99.)],dim=1)
        yp=torch.cat([y,torch.full((1,13,2),-99.)],dim=1).requires_grad_()
        valid=torch.tensor([[True]*3+[False]*13])
        ed=raw_energy_distance(xp,yp,valid=valid)
        torch.testing.assert_close(ed,raw_energy_distance(x,y))
        mmd,_=multiscale_rbf_mmd(xp,yp,bandwidth_scales=[.5,1.,2.,4.],valid=valid)
        ref,_=multiscale_rbf_mmd(x,y,bandwidth_scales=[.5,1.,2.,4.])
        torch.testing.assert_close(mmd,ref)
        mse=support_weighted_expression_mse(yp,xp,zero_weight=0.,valid=valid)
        torch.testing.assert_close(mse,support_weighted_expression_mse(y,x,zero_weight=0.))
        (ed+mmd+mse).sum().backward()
        self.assertTrue(torch.isfinite(yp.grad).all())
        self.assertEqual(yp.grad[:,3:].abs().sum().item(),0.)

    def test_generation_trims_padding_and_keeps_context(self):
        def sample(chunk,**kwargs):
            self.assertEqual(len(chunk['source']),16)
            self.assertEqual(len(torch.unique(chunk['batch_idx'])),1)
            return chunk['source']+10
        dummy=SimpleNamespace(population_set_size=16,_sample_population=sample)
        for n in [1,15,16,17,37,600]:
            batch={'source':torch.arange(n,dtype=torch.float32).reshape(n,1),'batch_idx':torch.zeros(n,dtype=torch.long)}
            if n>2:batch['batch_idx'][-2:]=1
            output=scDVDModule.bridge_sample(dummy,batch)
            torch.testing.assert_close(output,batch['source']+10)
            self.assertEqual(len(output),n)

if __name__=='__main__':unittest.main()
