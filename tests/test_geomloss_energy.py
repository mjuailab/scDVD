import unittest
import torch
from geomloss import SamplesLoss
from scdvd.module import geomloss_energy_distance, raw_energy_distance


class GeomLossEnergyTest(unittest.TestCase):
    def test_matches_perturbdiff_value_and_gradient(self):
        torch.manual_seed(123)
        x = torch.rand(3, 16, 7)
        y = torch.rand(3, 16, 7, requires_grad=True)
        actual = geomloss_energy_distance(x, y)
        expected = SamplesLoss(loss="energy", blur=0.05)(x, y)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), y, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), y)[0])
        # GeomLoss clamps squared distances for numerical stability at zero.
        torch.testing.assert_close(actual, raw_energy_distance(x, y) / 2, atol=2e-4, rtol=2e-4)

    def test_padding_equals_unpadded_and_has_no_direct_gradient(self):
        torch.manual_seed(321)
        x = torch.rand(2, 16, 7)
        y = torch.rand(2, 16, 7, requires_grad=True)
        valid = torch.arange(16).unsqueeze(0) < torch.tensor([[3], [11]])
        actual = geomloss_energy_distance(x, y, valid=valid)
        expected = torch.stack([SamplesLoss(loss="energy", blur=0.05)(x[i, valid[i]], y[i, valid[i]]) for i in range(2)])
        # Batched and unbatched matrix products can differ near zero distance.
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertEqual(y.grad[~valid].abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
