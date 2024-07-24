import unittest
import numpy as np
from src.data.data_transforms import make_atom_features


class TestAtomFeatures(unittest.TestCase):
    def setUp(self):
        self.protein = {
            "aatype": np.random.randint(0, 21, size=(100,)),
        }

    def test_forward(self):
        atom_features = make_atom_features(self.protein)
        self.assertEqual(atom_features["ref_pos"].shape, (100 * 4, 3))
        self.assertEqual(atom_features["ref_mask"].shape, (100 * 4,))
        self.assertEqual(atom_features["ref_element"].shape, (100 * 4, 4))
        self.assertEqual(atom_features["ref_charge"].shape, (100 * 4, ))
        self.assertEqual(atom_features["ref_atom_name_chars"].shape, (100 * 4, 3))
        self.assertEqual(atom_features["ref_space_uid"].shape, (100 * 4, 4))
