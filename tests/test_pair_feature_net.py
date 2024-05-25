import unittest
import torch
from prettytable import PrettyTable

from tests.config import consts
from src.models.feature_net import PairFeatureNet


class TestPairFeatureNet(unittest.TestCase):
    def test_shape(self):
        c_z = 128
        featurizer = PairFeatureNet(c_z=c_z)
        batch_size = 1
        n_res = 32

        ca_coordinates = torch.rand((batch_size, n_res, 3))
        residue_index = torch.arange(n_res).unsqueeze(0)
        residue_mask = torch.ones((batch_size, n_res))

        output = featurizer(residue_index, ca_coordinates, residue_mask)
        shape_after = output.shape

        self.assertTrue((batch_size, n_res, n_res, c_z) == shape_after)


if __name__ == '__main__':
    unittest.main()
