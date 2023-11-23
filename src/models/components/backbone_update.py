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

import torch
from torch import nn

from src.models.components.primitives import Linear
from src.utils.rigid_utils import Rigids, Rotations


class BackboneUpdate(nn.Module):
    """
        Implements Algorithm 23.
    """

    def __init__(self, c_s):
        """
            Args:
                c_s:
                    Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6)  # , init="final")

    def forward(self, s):
        """
            Args:
                [*, N_res, C_s] single representation
            Returns:
                [*, N_res] affine transformation object
        """
        # [*, 6]
        params = self.linear(s)

        # [*, 3]
        quats, trans = params[..., :3], params[..., 3:]

        # [*]
        norm_denominator = torch.sqrt(torch.sum(quats ** 2, dim=-1) + 1)

        # As many ones as there are dimensions in quats
        ones = s.new_ones((1,) * len(quats.shape))

        # [*, 4]
        quats = torch.cat((ones.expand(*quats.shape[:-1], 1), quats), dim=-1)
        quats = quats / norm_denominator.unsqueeze(-1)

        # [*, 3, 3]
        rots = Rotations(quats=quats)

        return Rigids(rots, trans)
