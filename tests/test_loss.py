import unittest
import torch
from src.diffusion.loss import mean_squared_error
from src.utils.geometry.vector import Vec3Array


class TestMeanSquaredError(unittest.TestCase):

    def test_basic_functionality(self):
        # Test MSE computation without weights or mask
        pred_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]]]))
        weights = torch.ones((1, 2))

        expected_mse = torch.mean(torch.sum((pred_atoms.to_tensor() - gt_atoms.to_tensor())**2, dim=-1) / 3, dim=1)
        mse = mean_squared_error(pred_atoms, gt_atoms, weights)
        self.assertTrue(torch.allclose(mse, expected_mse), f"Expected MSE {expected_mse}, got {mse}")

    def test_weighted_mse(self):
        # Test MSE with weights
        pred_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]]]))
        weights = torch.tensor([[0.5, 2.0]])  # Increased weight on the second atom

        mse_first_atom = torch.sum((pred_atoms.to_tensor()[0, 0, :] - gt_atoms.to_tensor()[0, 0, :]) ** 2, dim=-1)
        mse_second_atom = torch.sum((pred_atoms.to_tensor()[0, 1, :] - gt_atoms.to_tensor()[0, 1, :]) ** 2, dim=-1)

        mse_first_atom = mse_first_atom * weights[0, 0]
        mse_second_atom = mse_second_atom * weights[0, 1]

        expected_mse = (mse_first_atom + mse_second_atom) / 2.0 / 3.0  # Weighted MSE
        mse = mean_squared_error(pred_atoms, gt_atoms, weights)
        self.assertTrue(torch.allclose(mse, expected_mse), f"Expected MSE {expected_mse}, got {mse}")

    def test_mse_with_mask(self):
        # Test MSE computation with mask
        pred_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[1.0, 2.0, 3.0], [4.0, 8.0, 12.0]]]))
        weights = torch.tensor([[0.5, 2.0]])  # Increased weight on the second atom
        mask = torch.tensor([[1, 0]])  # Mask out the second atom

        expected_mse = torch.tensor([0.0])  # No error should be considered from the second atom
        mse = mean_squared_error(pred_atoms, gt_atoms, weights, mask)
        self.assertTrue(torch.allclose(mse, expected_mse), f"Expected MSE {expected_mse}, got {mse}")


if __name__ == '__main__':
    unittest.main()
