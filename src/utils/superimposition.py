from src.utils.geometry.alignment import weighted_rigid_align
from src.utils.geometry.vector import Vec3Array
import torch


def compute_rmsd(tensor1, tensor2, eps=1e-6):
    """Compute the RMSD between two tensors."""
    diff = tensor1 - tensor2
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum(dim=-1)
    mean_squared_diff = sum_squared_diff.mean(dim=-1)
    rmsd = torch.sqrt(mean_squared_diff + eps)
    return rmsd


def superimpose(reference, coords, mask):
    """
        Superimposes coordinates onto a reference by minimizing RMSD using SVD.

        Args:
            reference:
                [*, N, 3] reference tensor
            coords:
                [*, N, 3] tensor
            mask:
                [*, N] tensor
        Returns:
            A tuple of [*, N, 3] superimposed coords and [*] final RMSDs.
    """
    # To Vec3Array for alignment
    reference = Vec3Array.from_array(reference)
    coords = Vec3Array.from_array(coords)

    # Align the coordinates to the reference
    aligned_coords = weighted_rigid_align(coords, reference, weights=mask, mask=mask)
    aligned_coords = aligned_coords.to_tensor()

    # Compute RMSD
    rmsds = compute_rmsd(reference, aligned_coords)
    return aligned_coords, rmsds
