# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""Backbone update with the geometry package."""

import torch
import torch.nn as nn

from src.models.components.primitives import Linear
from src.utils.geometry.rigid_matrix_vector import Rigid3Array
from src.utils.geometry.rotation_matrix import Rot3Array
from src.utils.geometry.vector import Vec3Array


class BackboneUpdate(nn.Module):
    """Computes the backbone update."""

    def __init__(
            self,
            c_hidden,
            full_quat) -> None:
        """
        Args:
            c_hidden: The number of hidden channels.
            full_quat: Whether to use full quaternion representation.
        """
        super().__init__()
        self.full_quat = full_quat
        if self.full_quat:
            rigid_dim = 7
        else:
            rigid_dim = 6

        self.linear = Linear(c_hidden, rigid_dim, init="final", precision=torch.float32)

    def forward(self, activations: torch.Tensor) -> Rigid3Array:
        # NOTE: During training, this needs to be run in higher precision
        rigid_flat = self.linear(activations)

        rigid_flat = torch.unbind(rigid_flat, dim=-1)
        if self.full_quat:
            qw, qx, qy, qz = rigid_flat[:4]
            translation = rigid_flat[4:]
        else:
            qx, qy, qz = rigid_flat[:3]
            qw = torch.ones_like(qx)
            translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(
            qw, qx, qy, qz, normalize=True,
        )
        translation = Vec3Array(*translation)
        return Rigid3Array(rotation, translation)
