"""AlphaFold3 losses."""

import torch
from torch import Tensor
from src.utils.geometry.vector import Vec3Array, square_euclidean_distance, euclidean_distance
from src.utils.geometry.alignment import weighted_rigid_align
from torch.nn import functional as F
from typing import Optional, Set, Tuple
from src.utils.tensor_utils import one_hot


def smooth_lddt_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        atom_is_rna: Tensor,  # (bs, n_atoms)
        atom_is_dna: Tensor,  # (bs, n_atoms)
        mask: Tensor = None  # (bs, n_atoms)
) -> Tensor:  # (bs,)
    """Smooth local distance difference test (LDDT) auxiliary loss."""
    bs, n_atoms = pred_atoms.shape

    # Compute distances between all pairs of atoms
    delta_x_lm = euclidean_distance(pred_atoms[:, :, None], pred_atoms[:, None, :])  # (bs, n_atoms, n_atoms)
    delta_x_gt_lm = euclidean_distance(gt_atoms[:, :, None], gt_atoms[:, None, :])

    # Compute distance difference for all pairs of atoms
    delta_lm = torch.abs(delta_x_gt_lm - delta_x_lm)  # (bs, n_atoms, n_atoms)
    epsilon_lm = torch.div((F.sigmoid(torch.sub(0.5, delta_lm)) + F.sigmoid(torch.sub(1.0, delta_lm)) +
                            F.sigmoid(torch.sub(2.0, delta_lm)) + F.sigmoid(torch.sub(4.0, delta_lm))),
                           4.0)

    # Restrict to bespoke inclusion radius
    atom_is_nucleotide = (atom_is_dna + atom_is_rna).unsqueeze(-1).expand_as(delta_x_gt_lm)
    atom_not_nucleotide = torch.add(torch.neg(atom_is_nucleotide), 1.0)  # (1 - atom_is_nucleotide)
    c_lm = (delta_x_gt_lm < 30.0).float() * atom_is_nucleotide + (delta_x_gt_lm < 15.0).float() * atom_not_nucleotide

    # Mask positions
    if mask is not None:
        c_lm = c_lm * (mask[:, :, None] * mask[:, None, :])

    # Compute mean, avoiding self-term
    self_mask = torch.eye(n_atoms).unsqueeze(0).expand_as(c_lm).to(c_lm.device)  # (bs, n_atoms, n_atoms)
    self_mask = torch.add(torch.neg(self_mask), 1.0)
    c_lm = c_lm * self_mask
    lddt = torch.mean(epsilon_lm * c_lm, dim=(1, 2)) / torch.mean(c_lm, dim=(1, 2))
    return torch.add(torch.neg(lddt), 1.0)  # (1 - lddt)


def mean_squared_error(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        weights: Tensor,  # (bs, n_atoms)
        mask: Tensor = None,  # (bs, n_atoms)
        epsilon: Optional[float] = 1e-5
) -> Tensor:  # (bs,)
    """Weighted MSE loss as the main training objective for diffusion."""

    # Compute atom-wise MSE
    atom_mse = square_euclidean_distance(pred_atoms, gt_atoms, epsilon=epsilon)  # (bs, n_atoms)

    # Mask positions
    if mask is not None:
        atom_mse *= mask

    # Weighted mean
    mse = torch.mean(atom_mse * weights, dim=1)
    return torch.div(mse, 3.0)


def bond_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        atom_indices: Set[Tuple[int, int]]
) -> Tensor:
    """Loss to ensure that the bons for bonded ligands (including bonded glycans)
    have the correct length."""
    raise NotImplementedError("the implementation of this function will depend on the input pipeline")


def diffusion_loss(
        pred_atoms: Vec3Array,  # (bs, n_atoms)
        gt_atoms: Vec3Array,  # (bs, n_atoms)
        timesteps: Tensor,  # (bs, 1)
        atom_is_rna: Tensor,  # (bs, n_atoms)
        atom_is_dna: Tensor,  # (bs, n_atoms)
        weights: Tensor,  # (bs, n_atoms)
        mask: Optional[Tensor] = None,  # (bs, n_atoms)
        sd_data: float = 16.0,  # Standard deviation of the data
        use_smooth_lddt: bool = False
) -> Tensor:  # (bs,)
    """Diffusion loss that scales the MSE and LDDT losses by the noise level (timestep).
    Args:
        pred_atoms:
        gt_atoms:
        timesteps:
        atom_is_rna:
        atom_is_dna:
        weights:
        mask:
        sd_data:
        use_smooth_lddt:
    """
    # Align the gt_atoms to pred_atoms
    aligned_gt_atoms = weighted_rigid_align(x=gt_atoms, x_gt=pred_atoms, weights=weights, mask=mask)

    # MSE loss
    mse = mean_squared_error(pred_atoms, aligned_gt_atoms, weights, mask)

    # Scale by (t**2 + σ**2) / (t + σ)**2
    scaling_factor = torch.add(timesteps ** 2, sd_data ** 2) / (torch.add(timesteps, sd_data) ** 2)
    loss_diffusion = scaling_factor * mse

    # Smooth LDDT Loss
    if use_smooth_lddt:
        lddt_loss = smooth_lddt_loss(pred_atoms, aligned_gt_atoms, atom_is_rna, atom_is_dna, mask)
        loss_diffusion = loss_diffusion + lddt_loss
    return loss_diffusion


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def distogram_loss(
        logits: Tensor,  # (bs, n_tokens, n_tokens, n_bins)
        pseudo_beta: Tensor,  # (bs, n_tokens, 3)
        pseudo_beta_mask: Tensor,  # (bs, n_tokens)
        min_bin: float = 2.0,
        max_bin: float = 22.0,
        no_bins: int = 64,
        eps: float = 1e-6,
        **kwargs,
) -> Tensor:  # (bs,)
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2
    dists = torch.sum(
        (pseudo_beta[..., :, None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdim=True,
    )
    true_bins = torch.sum(dists > boundaries, dim=-1)
    errors = softmax_cross_entropy(
        logits,
        F.one_hot(true_bins, no_bins),
    )
    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)
    return mean


def experimentally_resolved_loss(
        logits: Tensor,  # (bs, n_atoms, 2)
        atom_exists: Tensor,  # (bs, n_atoms)
        atom_mask: Tensor,  # (bs, n_atoms)
        eps: float = 1e-8,
        **kwargs,
) -> Tensor:  # (bs,)
    """Loss for training the experimentally resolved head. """
    is_resolved = F.one_hot(atom_exists.long(), num_classes=2).to(logits.dtype)  # (bs, n_atoms, 2)
    errors = softmax_cross_entropy(logits, is_resolved)  # (bs, n_atoms)
    loss = torch.sum(errors * atom_mask, dim=-1)
    loss = torch.sum(loss, dim=-1) / (eps + torch.sum(atom_mask, dim=-1, keepdim=True))
    return loss


def plddt_loss(
        logits: Tensor
) -> Tensor:
    # TODO: tricky to compute this

    # Compute d_lm
    # d_lm  (bs, n_tokens, n_tokens, N_max_atoms_per_token)

    #
    pass


def predicted_distance_error_loss(
        logits: Tensor,  # (bs, n_tokens, n_tokens, 64)
        atom_repr_pred: Tensor,  # (bs, n_tokens, 3)
        atom_repr_gt: Tensor,  # (bs, n_tokens, 3)
        token_mask: Tensor,  # (bs, n_tokens)
        eps: float = 1e-8,
) -> Tensor:  # (bs,)
    # Compute the distance error
    distance_error = torch.sqrt(
        torch.sum((atom_repr_pred[..., None, :] - atom_repr_gt[..., None, :, :]) ** 2, dim=-1)
    )
    # Bin the distance error, from 0.0 to 32.0 Angstroms in 64 bins
    v_bins = torch.linspace(0.0, 32.0, steps=64, device=logits.device, dtype=logits.dtype)
    distance_error = one_hot(distance_error, v_bins)
    # Compute cross entropy loss
    errors = softmax_cross_entropy(logits, distance_error)  # (bs, n_tokens, n_tokens)
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    loss = torch.sum(errors * pair_mask, dim=(-1, -2)) / (eps + torch.sum(pair_mask, dim=(-1, -2)))
    return loss


def predicted_aligned_error_loss():
    pass
