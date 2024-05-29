"""Data augmentations applied prior to sampling from the diffusion trajectory."""

import torch


def centre_random_augmentation(
        atom_positions: torch.Tensor,  # (bs, n_atoms, 3)
        s_trans: float = 1.0,  # Translation scaling factor
) -> torch.Tensor:
    """Centers the atoms and applies random rotation and translation."""
    # Center the atoms

    # Sample random rotation

    # Apply
    pass
