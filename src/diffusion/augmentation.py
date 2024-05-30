"""Data augmentations applied prior to sampling from the diffusion trajectory."""
import torch
from src.utils.geometry.vector import Vec3Array
from src.utils.geometry.rotation_matrix import Rot3Array


def centre_random_augmentation(
        atom_positions: Vec3Array,  # (bs, n_atoms)
        s_trans: float = 1.0,  # Translation scaling factor
) -> Vec3Array:  # (bs, n_atoms)
    """Centers the atoms and applies random rotation and translation.
    Args:
        atom_positions:
            [*, n_atoms] vector of atom coordinates.
        s_trans:
            Scaling factor in Angstroms for the random translation sampled
            from a normal distribution.
    Returns:
        [*, n_atoms] vector of atom coordinates after augmentation.
    """
    batch_size, n_atoms = atom_positions.shape
    device = atom_positions.x.device

    # Center the atoms
    center = atom_positions.mean(dim=1, keepdim=True)
    atom_positions = atom_positions - center

    # Sample random rotation
    rots = Rot3Array.uniform_random((batch_size, 1), device)

    # Sample random translation from normal distribution
    trans = s_trans * Vec3Array.randn((batch_size, 1), device)

    # Apply
    atom_positions = rots.apply_to_point(atom_positions) + trans
    return atom_positions
