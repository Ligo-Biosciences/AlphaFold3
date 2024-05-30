import unittest
import torch
from src.utils.geometry.vector import Vec3Array
from src.utils.geometry.rotation_matrix import Rot3Array
from src.diffusion.augmentation import centre_random_augmentation


class TestCentreRandomAugmentation(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_atoms = 5
        atom_positions = torch.randn(self.batch_size, self.n_atoms, 3)
        self.atom_positions = Vec3Array(x=atom_positions[:, :, 0],
                                        y=atom_positions[:, :, 1],
                                        z=atom_positions[:, :, 2])

    def test_output_shape(self):
        augmented_positions = centre_random_augmentation(self.atom_positions)
        self.assertEqual(augmented_positions.to_tensor().shape, (self.batch_size, self.n_atoms, 3))

    def test_random_translation(self):
        s_trans = 1.0
        initial_positions = self.atom_positions.x.clone()
        augmented_positions = centre_random_augmentation(self.atom_positions, s_trans=s_trans)
        translation = augmented_positions.x - initial_positions
        translation_magnitudes = translation.norm(dim=-1)
        self.assertTrue(torch.all(translation_magnitudes > 0))

    def test_random_rotation(self):
        initial_positions = self.atom_positions.x.clone()
        augmented_positions = centre_random_augmentation(self.atom_positions)
        rotation_diff = augmented_positions.x - initial_positions
        self.assertFalse(torch.allclose(rotation_diff, torch.zeros_like(rotation_diff), atol=1e-6))
