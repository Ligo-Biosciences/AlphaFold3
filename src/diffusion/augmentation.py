"""Data augmentations applied prior to sampling from the diffusion trajectory."""
from torch import Tensor
from src.utils.geometry.vector import Vec3Array
from src.utils.geometry.rotation_matrix import Rot3Array


def centre_random_augmentation(
        atom_positions: Vec3Array,  # (*, n_atoms)
        atom_mask: Tensor,  # (*, n_atoms)
        s_trans: float = 1.0,  # Translation scaling factor
) -> Vec3Array:  # (*, n_atoms)
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
    batch_dims = atom_positions.shape[:-1]
    device = atom_positions.x.device

    # Center the atoms
    center = atom_positions.sum(dim=-1, keepdim=True) / atom_mask.sum(dim=-1, keepdim=True)
    atom_positions = atom_positions - center

    # Sample random rotation
    rots = Rot3Array.uniform_random((batch_dims + (1,)), device)

    # Sample random translation from normal distribution
    trans = s_trans * Vec3Array.randn((batch_dims + (1,)), device)

    # Apply
    atom_positions = rots.apply_to_point(atom_positions) + trans

    # Apply mask
    atom_positions = atom_positions * atom_mask
    return atom_positions
