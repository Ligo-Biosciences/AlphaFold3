import torch
import unittest
from src.utils.geometry.alignment import weighted_rigid_align, compute_covariance_matrix
from src.utils.geometry.vector import Vec3Array
from src.utils.geometry.rotation_matrix import Rot3Array
from src.diffusion.augmentation import centre_random_augmentation


class TestAlignment(unittest.TestCase):

    def test_compute_covariance_matrix(self):
        # Test simple case
        P = torch.tensor([[[1, 2, 3]]], dtype=torch.float32)  # Shape (1, 1, 3)
        Q = torch.tensor([[[4, 5, 6]]], dtype=torch.float32)  # Shape (1, 1, 3)
        expected_H = torch.tensor([[[4., 5., 6.],
                                    [8., 10., 12.],
                                    [12., 15., 18.]]])  # Shape (1, 3, 3)
        H = compute_covariance_matrix(P, Q)

        self.assertTrue(torch.allclose(H, expected_H), "Covariance matrix calculation failed")

        # Test zero matrices
        P_zero = torch.zeros((1, 10, 3), dtype=torch.float32)
        Q_zero = torch.zeros((1, 10, 3), dtype=torch.float32)
        H_zero = compute_covariance_matrix(P_zero, Q_zero)
        self.assertTrue(torch.equal(H_zero, torch.zeros((1, 3, 3))), "Covariance matrix should be zero")

    def test_weighted_rigid_align(self):
        # Setup
        bs, n_atoms = 1, 3
        x = Vec3Array.from_array(torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]))
        x_gt = Vec3Array.from_array(torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]))
        weights = torch.ones((bs, n_atoms))
        mask = torch.ones((bs, n_atoms))

        # Identity alignment (no movement expected)
        x_aligned = weighted_rigid_align(x, x_gt, weights, mask)
        self.assertTrue(torch.allclose(x_aligned.to_tensor(), x.to_tensor(), atol=1e-6), "Alignment should be identity")

        # Rotation removal test
        n_atoms = 100
        x = Vec3Array.from_array(torch.randn((bs, n_atoms, 3)))
        x = x - x.mean(dim=1, keepdim=True)  # Center x
        weights = torch.ones((bs, n_atoms))
        mask = torch.ones((bs, n_atoms))

        x_rotated = centre_random_augmentation(x, s_trans=0.0)
        x_rotated_aligned = weighted_rigid_align(x_rotated, x, weights, mask)
        self.assertTrue(torch.allclose(x_rotated_aligned.to_tensor(), x.to_tensor(), atol=1e-6), "Rotation should be "
                                                                                                 "corrected")


if __name__ == '__main__':
    unittest.main()
