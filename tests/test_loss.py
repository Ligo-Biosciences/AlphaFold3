import unittest
import torch
from src.utils.loss import mean_squared_error, smooth_lddt_loss, diffusion_loss
from src.diffusion.sample import sample_noise_level, noise_positions
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


class TestSmoothLDDTLoss(unittest.TestCase):

    def test_basic_functionality(self):
        # Create a simple scenario where pred_atoms and gt_atoms are the same, no nucleotide-specific settings.
        bs, n_atoms = 1, 4
        pred_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        atom_is_rna = torch.zeros((bs, n_atoms))
        atom_is_dna = torch.zeros((bs, n_atoms))
        mask = None

        # Expected loss is 0 since pred_atoms == gt_atoms
        loss_when_identical = smooth_lddt_loss(pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)

        # Add noise and compare
        noisy_pred_atoms = pred_atoms + 0.1 * Vec3Array.from_array(torch.randn((bs, n_atoms, 3)))
        noisier_pred_atoms = pred_atoms + 1.0 * Vec3Array.from_array(torch.randn((bs, n_atoms, 3)))

        loss_when_noisy = smooth_lddt_loss(noisy_pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)
        loss_when_noisier = smooth_lddt_loss(noisier_pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)

        self.assertTrue(torch.all((loss_when_identical < loss_when_noisy) & (loss_when_noisy < loss_when_noisier)))

    def test_mask(self):
        # Create a simple scenario where pred_atoms and gt_atoms are the same, no nucleotide-specific settings.
        bs, n_atoms = 1, 4
        pred_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        atom_is_rna = torch.randint(0, 2, (bs, n_atoms))
        atom_is_dna = torch.zeros((bs, n_atoms))
        mask = torch.ones((bs, n_atoms))

        # Expected loss is 0 since pred_atoms == gt_atoms
        loss_when_identical = smooth_lddt_loss(pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)

        # Add noise and compare
        noisy_pred_atoms = pred_atoms + 5.0 * Vec3Array.from_array(torch.randn((bs, n_atoms, 3)))
        noisier_pred_atoms = pred_atoms + 10.0 * Vec3Array.from_array(torch.randn((bs, n_atoms, 3)))

        loss_when_noisy = smooth_lddt_loss(noisy_pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)
        loss_when_noisier = smooth_lddt_loss(noisier_pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)

        self.assertTrue(torch.all((loss_when_identical < loss_when_noisy) & (loss_when_noisy < loss_when_noisier)))


class TestDiffusionLoss(unittest.TestCase):
    def test_basic_functionality(self):
        # Create a simple scenario where pred_atoms and gt_atoms are the same, no nucleotide-specific settings.
        bs, n_atoms = 1, 4
        pred_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        gt_atoms = Vec3Array.from_array(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]))
        atom_is_rna = torch.zeros((bs, n_atoms))
        atom_is_dna = torch.zeros((bs, n_atoms))
        timesteps = sample_noise_level(torch.randn(bs, 1))
        weights = torch.ones((bs, n_atoms))
        mask = None

        loss_when_identical = diffusion_loss(pred_atoms, gt_atoms, timesteps, atom_is_rna, atom_is_dna, weights, mask)

        # Add noise and compare
        noisy_pred_atoms = noise_positions(pred_atoms, timesteps)
        noisier_pred_atoms = noise_positions(noisy_pred_atoms, timesteps)

        loss_when_noisy = diffusion_loss(noisy_pred_atoms, gt_atoms, timesteps, atom_is_rna, atom_is_dna, weights, mask)
        loss_when_noisier = diffusion_loss(noisier_pred_atoms, gt_atoms, timesteps, atom_is_rna, atom_is_dna, weights, mask)

        self.assertTrue(isinstance(loss_when_identical, torch.Tensor))
        self.assertTrue(torch.all((loss_when_identical < loss_when_noisy) & (loss_when_noisy < loss_when_noisier)))


if __name__ == '__main__':
    unittest.main()
