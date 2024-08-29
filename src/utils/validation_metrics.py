import torch
from typing import Optional


def drmsd(structure_1: torch.Tensor, structure_2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate the distance Root Mean Square Deviation (dRMSD) between two structures.

    Args:
        structure_1 (torch.Tensor): First structure of shape [..., N, 3]
        structure_2 (torch.Tensor): Second structure of shape [..., N, 3]
        mask (Optional[torch.Tensor]): Mask of shape [..., N] indicating valid positions

    Returns:
        torch.Tensor: The dRMSD between the two structures
    """
    def pairwise_distances(structure: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise distances within a structure."""
        diff = structure[..., :, None, :] - structure[..., None, :, :]
        return torch.norm(diff, dim=-1)

    d1 = pairwise_distances(structure_1)
    d2 = pairwise_distances(structure_2)

    squared_diff = (d1 - d2) ** 2

    if mask is not None:
        mask_2d = mask[..., None] * mask[..., None, :]
        squared_diff = squared_diff * mask_2d
        n = torch.sum(mask, dim=-1)
    else:
        n = d1.shape[-1]

    sum_squared_diff = torch.sum(squared_diff, dim=(-1, -2))

    # Avoid division by zero
    drmsd = torch.sqrt(sum_squared_diff / (n * (n - 1) + 1e-8))

    return drmsd


def drmsd_np(structure_1, structure_2, mask=None):
    structure_1 = torch.tensor(structure_1)
    structure_2 = torch.tensor(structure_2)
    if mask is not None:
        mask = torch.tensor(mask)

    return drmsd(structure_1, structure_2, mask)


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    """
    Compute the local distance difference test (LDDT) score.

    Algorithm steps:
    1. Compute pairwise distance matrices for true and predicted structures
    2. Identify pairs within the cutoff distance
    3. Calculate absolute differences between true and predicted distances
    4. Score the differences based on predefined thresholds
    5. Normalize the scores

    Args:
        all_atom_pred_pos: Predicted atom positions [..., N, 3]
        all_atom_positions: True atom positions [..., N, 3]
        all_atom_mask: Mask for valid atoms [..., N]
        cutoff: Distance cutoff for considering atom pairs
        eps: Small value to avoid division by zero
        per_residue: If True, return per-residue scores

    Returns:
        LDDT scores
    """
    # Compute pairwise distance matrices
    dmat_true = torch.cdist(all_atom_positions, all_atom_positions)
    dmat_pred = torch.cdist(all_atom_pred_pos, all_atom_pred_pos)

    # Create mask for pairs within cutoff
    dists_to_score = (dmat_true < cutoff).float()
    
    # Apply atom mask
    atom_mask_2d = all_atom_mask.unsqueeze(-1) * all_atom_mask.unsqueeze(-2)
    dists_to_score = dists_to_score * atom_mask_2d

    # Remove self-interactions
    eye = torch.eye(dists_to_score.shape[-1], device=dists_to_score.device)
    eye = eye.view(*([1] * (dists_to_score.dim() - 2)), *eye.shape)
    dists_to_score = dists_to_score * (1 - eye)

    # Calculate absolute differences
    dist_l1 = torch.abs(dmat_true - dmat_pred)

    # Score the differences
    score = torch.zeros_like(dist_l1)
    for threshold in [0.5, 1.0, 2.0, 4.0]:
        score += (dist_l1 < threshold).float()
    score *= 0.25  # Normalize by number of thresholds

    # Apply scoring mask and sum
    scored_pairs = dists_to_score * score

    if per_residue:
        # Compute per-residue scores
        residue_scores = scored_pairs.sum(dim=-1)
        residue_weights = dists_to_score.sum(dim=-1)
        lddt_score = residue_scores / (residue_weights + eps)
    else:
        # Compute global score
        lddt_score = scored_pairs.sum(dim=(-2, -1)) / (dists_to_score.sum(dim=(-2, -1)) + eps)

    return lddt_score

    
def gdt(p1, p2, mask, cutoffs):
    """
    Calculate the Global Distance Test (GDT) score for protein structures.

    Args:
        p1 (torch.Tensor): Coordinates of the first structure [..., N, 3].
        p2 (torch.Tensor): Coordinates of the second structure [..., N, 3].
        mask (torch.Tensor): Mask for valid residues [..., N].
        cutoffs (list): List of distance cutoffs for GDT calculation.

    Returns:
        torch.Tensor: GDT score [...].
    """
    # Ensure inputs are float
    p1 = p1.float()
    p2 = p2.float()
    mask = mask.float()

    # Calculate number of valid residues per batch
    n = torch.sum(mask, dim=-1)

    # Calculate pairwise distances
    distances = torch.sqrt(torch.sum((p1 - p2)**2, dim=-1))

    scores = []
    for c in cutoffs:
        # Calculate score for each cutoff, accounting for the mask
        score = torch.sum((distances <= c).float() * mask, dim=-1) / (n + 1e-8)
        scores.append(score)

    # Stack scores and average across cutoffs
    scores = torch.stack(scores, dim=-1)
    return torch.mean(scores, dim=-1)


def gdt_ts(p1, p2, mask):
    """
    Calculate the Global Distance Test Total Score (GDT_TS).

    Args:
        p1 (torch.Tensor): Coordinates of the first structure [..., N, 3].
        p2 (torch.Tensor): Coordinates of the second structure [..., N, 3].
        mask (torch.Tensor): Mask for valid residues [..., N].

    Returns:
        torch.Tensor: GDT_TS score [...].
    """
    return gdt(p1, p2, mask, [1., 2., 4., 8.])


def gdt_ha(p1, p2, mask):
    """
    Calculate the Global Distance Test High Accuracy (GDT_HA) score.

    Args:
        p1 (torch.Tensor): Coordinates of the first structure [..., N, 3].
        p2 (torch.Tensor): Coordinates of the second structure [..., N, 3].
        mask (torch.Tensor): Mask for valid residues [..., N].

    Returns:
        torch.Tensor: GDT_HA score [...].
    """
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])