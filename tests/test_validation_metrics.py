import unittest
import torch
import numpy as np
from src.utils.validation_metrics import drmsd, gdt_ts, gdt_ha, lddt
from typing import Optional
from Bio.PDB import PDBParser
import random


class TestValidationMetricsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the PDB file
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("1bn1", "tests/test_data/1bn1.pdb")
        
        # Extract coordinates and create a mask
        coords = []
        mask = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):  # Only consider CA atoms
                        coords.append(residue["CA"].coord)
                        mask.append(1)
                    else:
                        mask.append(0)
        
        # Create a batch of 3 identical structures
        cls.coords = torch.tensor(coords).float().unsqueeze(0).repeat(3, 1, 1)
        cls.mask = torch.tensor(mask).float().unsqueeze(0).repeat(3, 1)

    def test_print_shapes(self):
        print("Shape of coords:", self.coords.shape)
        print("Shape of mask:", self.mask.shape)


class TestDRMSD(TestValidationMetricsBase):
    def test_drmsd_shape(self):
        result = drmsd(self.coords, self.coords, self.mask)
        self.assertEqual(result.shape, torch.Size([3]))

    def test_drmsd_identical_structures(self):
        result = drmsd(self.coords, self.coords, self.mask)
        self.assertTrue(torch.allclose(result, torch.zeros(3), atol=1e-6))

    def test_drmsd_noisy_structures(self):
        noisy_coords = self.coords + torch.randn_like(self.coords) * 0.1
        result = drmsd(self.coords, noisy_coords, self.mask)
        self.assertTrue(torch.all(result > 0.0))

    def test_drmsd_increases_with_noise(self):
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        drmsd_scores = []
        
        for noise in noise_levels:
            noisy_coords = self.coords + torch.randn_like(self.coords) * noise
            drmsd_result = drmsd(self.coords, noisy_coords, self.mask)
            drmsd_scores.append(drmsd_result.mean().item())
        
        # Check if dRMSD scores are increasing
        for i in range(1, len(drmsd_scores)):
            self.assertGreater(drmsd_scores[i], drmsd_scores[i-1], 
                               f"dRMSD score did not increase from noise level {noise_levels[i-1]} to {noise_levels[i]}")
        
        print("dRMSD scores for increasing noise levels:")
        for noise, score in zip(noise_levels, drmsd_scores):
            print(f"Noise: {noise}, dRMSD: {score:.4f}")


class TestLDDT(TestValidationMetricsBase):
    def test_lddt_shape(self):
        result = lddt(self.coords, self.coords, self.mask)
        self.assertEqual(result.shape, self.mask.shape)

    def test_lddt_identical_structures(self):
        result = lddt(self.coords, self.coords, self.mask)
        self.assertTrue(torch.allclose(result, torch.ones_like(result)))

    def test_lddt_noisy_structures(self):
        noisy_coords = self.coords + torch.randn_like(self.coords) * 0.5
        result = lddt(self.coords, noisy_coords, self.mask)
        self.assertTrue(torch.all(result <= 1.0))
        self.assertTrue(torch.all(result >= 0.0))
    
    def test_lddt_decreases_with_noise(self):
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        lddt_scores = []
        
        for noise in noise_levels:
            noisy_coords = self.coords + torch.randn_like(self.coords) * noise
            lddt_result = lddt(self.coords, noisy_coords, self.mask)
            lddt_scores.append(lddt_result.mean().item())
        
        # Check if LDDT scores are decreasing
        for i in range(1, len(lddt_scores)):
            self.assertLess(lddt_scores[i], lddt_scores[i-1], 
                            f"LDDT score did not decrease from noise level {noise_levels[i-1]} to {noise_levels[i]}")
        
        print("LDDT scores for increasing noise levels:")
        for noise, score in zip(noise_levels, lddt_scores):
            print(f"Noise: {noise}, LDDT: {score:.4f}")


class TestGDTMetrics(TestValidationMetricsBase):
    def test_gdt_ts_shape(self):
        result = gdt_ts(self.coords, self.coords, self.mask)
        self.assertEqual(result.shape, torch.Size([3]))

    def test_gdt_ts_identical_structures(self):
        result = gdt_ts(self.coords, self.coords, self.mask)
        self.assertTrue(torch.allclose(result, torch.ones(3), atol=1e-6))

    def test_gdt_ts_noisy_structures(self):
        noisy_coords = self.coords + torch.randn_like(self.coords) * 0.5
        result = gdt_ts(self.coords, noisy_coords, self.mask)
        self.assertTrue(torch.all(result < 1.0))
        self.assertTrue(torch.all(result > 0.0))

    def test_gdt_ha_shape(self):
        result = gdt_ha(self.coords, self.coords, self.mask)
        self.assertEqual(result.shape, torch.Size([3]))

    def test_gdt_ha_identical_structures(self):
        result = gdt_ha(self.coords, self.coords, self.mask)
        self.assertTrue(torch.allclose(result, torch.ones(3), atol=1e-6))

    def test_gdt_ha_noisy_structures(self):
        noisy_coords = self.coords + torch.randn_like(self.coords) * 0.5
        result = gdt_ha(self.coords, noisy_coords, self.mask)
        self.assertTrue(torch.all(result < 1.0))
        self.assertTrue(torch.all(result > 0.0))

    def test_metrics_with_increasing_noise(self):
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        for noise in noise_levels:
            noisy_coords = self.coords + torch.randn_like(self.coords) * noise
            gdt_ts_result = gdt_ts(self.coords, noisy_coords, self.mask)
            gdt_ha_result = gdt_ha(self.coords, noisy_coords, self.mask)
            
            print(f"Noise level: {noise}")
            print(f"GDT-TS: {gdt_ts_result.mean().item():.4f}")
            print(f"GDT-HA: {gdt_ha_result.mean().item():.4f}")
            print()

    def test_metrics_with_random_perturbations(self):
        def perturb_structure(coords, mask, fraction=0.1, max_displacement=2.0):
            perturbed_coords = coords.clone()
            num_residues = mask.sum(dim=1).int()
            for i in range(coords.shape[0]):
                num_to_perturb = int(fraction * num_residues[i].item())
                indices_to_perturb = random.sample(range(num_residues[i].item()), num_to_perturb)
                
                for idx in indices_to_perturb:
                    displacement = torch.randn(3) * max_displacement
                    perturbed_coords[i, idx] += displacement
            
            return perturbed_coords

        fractions = [0.1, 0.2, 0.5, 0.8]
        for fraction in fractions:
            perturbed_coords = perturb_structure(self.coords, self.mask, fraction)
            gdt_ts_result = gdt_ts(self.coords, perturbed_coords, self.mask)
            gdt_ha_result = gdt_ha(self.coords, perturbed_coords, self.mask)
            
            print(f"Fraction perturbed: {fraction}")
            print(f"GDT-TS: {gdt_ts_result.mean().item():.4f}")
            print(f"GDT-HA: {gdt_ha_result.mean().item():.4f}")
            print()


if __name__ == '__main__':
    unittest.main()