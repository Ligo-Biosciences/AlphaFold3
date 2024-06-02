"""Diffusion losses"""

import torch
from src.utils.geometry.vector import Vec3Array, square_euclidean_distance
from torch import nn


def smooth_lddt_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        atom_is_rna: torch.Tensor,  # (bs, n_atoms)
        atom_is_dna: torch.Tensor,  # (bs, n_atoms)
        mask: torch.Tensor = None  # (bs, n_atoms)
) -> torch.Tensor:  # (bs,)
    """Smooth local distance difference test (LDDT) auxiliary loss."""
    raise NotImplementedError


def mean_squared_error(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        weights: torch.Tensor,  # (bs, n_atoms)
        mask: torch.Tensor = None  # (bs, n_atoms)
) -> torch.Tensor:  # (bs,)
    """Weighted MSE loss as the main training objective for diffusion."""

    # Compute atom-wise MSE
    atom_mse = square_euclidean_distance(pred_atoms, gt_atoms, epsilon=None)  # (bs, n_atoms)

    # Mask positions
    if mask is not None:
        atom_mse *= mask

    # Weighted mean
    mse = torch.mean(atom_mse * weights, dim=1)
    return torch.div(mse, 3.0)
