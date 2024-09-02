# Copyright 2024 Ligo Biosciences Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        atom_mask:
            [*, n_atoms] mask of which atoms are valid (non-padding).
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
