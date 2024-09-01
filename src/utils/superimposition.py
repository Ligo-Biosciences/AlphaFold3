from src.utils.geometry.alignment import weighted_rigid_align
from src.utils.geometry.vector import Vec3Array
import torch


def compute_rmsd(tensor1, tensor2, mask, eps=1e-6):
    """Compute the RMSD between two tensors."""
    diff = tensor1 - tensor2
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum(dim=-1)

    # Mask out invalid positions
    sum_squared_diff = sum_squared_diff * mask

    # Average over valid positions
    denom = mask.sum(dim=-1) + eps
    mean_squared_diff = torch.sum(sum_squared_diff, dim=-1) / denom

    # Square root to get RMSD
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
    reference = reference.to_tensor()
    # Compute RMSD
    rmsds = compute_rmsd(reference, aligned_coords)
    return aligned_coords, rmsds
