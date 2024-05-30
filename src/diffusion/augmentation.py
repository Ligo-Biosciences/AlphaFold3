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
    center = atom_positions.mean(dim=-1, keepdim=True)
    atom_positions = atom_positions - center

    # Sample random rotation
    quaternions = torch.randn(batch_size, 4, device=device)
    rots = Rot3Array.from_quaternion(w=quaternions[:, 0],
                                     x=quaternions[:, 1],
                                     y=quaternions[:, 2],
                                     z=quaternions[:, 3],
                                     normalize=True)  # (bs)
    rots = rots.unsqueeze(-1)  # (bs, 1)

    # Sample random translation
    trans = s_trans * Vec3Array.from_array(torch.randn((batch_size, 3), device=device))
    trans = trans.unsqueeze(-1)  # (bs, 1)

    # Apply
    atom_positions = rots.apply_to_point(atom_positions) + trans
    return atom_positions
