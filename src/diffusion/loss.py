"""Diffusion losses."""

import torch
from src.utils.geometry.vector import Vec3Array, square_euclidean_distance, euclidean_distance
from torch.nn import functional as F
from typing import Optional


def smooth_lddt_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        atom_is_rna: torch.Tensor,  # (bs, n_atoms)
        atom_is_dna: torch.Tensor,  # (bs, n_atoms)
        mask: torch.Tensor = None  # (bs, n_atoms)
) -> torch.Tensor:  # (bs,)
    """Smooth local distance difference test (LDDT) auxiliary loss."""
    bs, n_atoms = pred_atoms.shape

    # Compute distances between all pairs of atoms
    delta_x_lm = euclidean_distance(pred_atoms[:, :, None], pred_atoms[:, None, :])  # (bs, n_atoms, n_atoms)
    delta_x_gt_lm = euclidean_distance(gt_atoms[:, :, None], gt_atoms[:, None, :])

    # Compute distance difference for all pairs of atoms
    delta_lm = torch.abs(delta_x_gt_lm - delta_x_lm)  # (bs, n_atoms, n_atoms)
    epsilon_lm = torch.div((F.sigmoid(torch.sub(0.5, delta_lm)) + F.sigmoid(torch.sub(1.0, delta_lm)) +
                            F.sigmoid(torch.sub(2.0, delta_lm)) + F.sigmoid(torch.sub(4.0, delta_lm))), 4.0)

    # Restrict to bespoke inclusion radius
    atom_is_nucleotide = (atom_is_dna + atom_is_rna).unsqueeze(-1).expand_as(delta_x_gt_lm)
    atom_not_nucleotide = torch.add(torch.neg(atom_is_nucleotide), 1.0)  # (1 - atom_is_nucleotide)
    c_lm = (delta_x_gt_lm < 30.0).float() * atom_is_nucleotide + (delta_x_gt_lm < 15.0).float() * atom_not_nucleotide

    # Mask positions
    if mask is not None:
        c_lm *= (mask[:, :, None] * mask[:, None, :])

    # Compute mean, avoiding self-term
    self_mask = torch.eye(n_atoms).unsqueeze(0).expand_as(c_lm).to(c_lm.device)  # (bs, n_atoms, n_atoms)
    self_mask = torch.add(torch.neg(self_mask), 1.0)
    c_lm *= self_mask
    lddt = torch.mean(epsilon_lm * c_lm, dim=(1, 2)) / torch.mean(c_lm, dim=(1, 2))
    return torch.add(torch.neg(lddt), 1.0)  # (1 - lddt)


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


def diffusion_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        timesteps: torch.Tensor,  # (bs, 1)
        atom_is_rna: torch.Tensor,  # (bs, n_atoms)
        atom_is_dna: torch.Tensor,  # (bs, n_atoms)
        weights: torch.Tensor,  # (bs, n_atoms)
        mask: Optional[torch.Tensor] = None,  # (bs, n_atoms)
        sd_data: float = 16.0,  # Standard deviation of the data
) -> torch.Tensor:  # (bs,)
    """Diffusion loss that scales the MSE and LDDT losses by the noise level (timestep)."""
    mse = mean_squared_error(pred_atoms, gt_atoms, weights, mask)
    lddt_loss = smooth_lddt_loss(pred_atoms, gt_atoms, atom_is_rna, atom_is_dna, mask)

    # Scale by (t**2 + σ**2) / (t + σ)**2
    scaling_factor = torch.add(timesteps ** 2, sd_data ** 2) / (torch.add(timesteps, sd_data) ** 2)
    loss_diffusion = scaling_factor * mse + lddt_loss
    return loss_diffusion
