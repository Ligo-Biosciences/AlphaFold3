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

"""Implements the Invariant Point Attention module."""

import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Sequence, Union

from src.utils.precision_utils import is_fp16_enabled
from src.utils.rigid_utils import Rotations, Rigids
from src.utils.geometry.rigid_matrix_vector import Rigid3Array
from src.utils.geometry.vector import Vec3Array, square_euclidean_distance


from src.models.components.primitives import Linear, ipa_point_weights_init_
from src.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class PointProjection(nn.Module):
    def __init__(self,
                 c_hidden: int,
                 num_points: int,
                 no_heads: int,
                 is_multimer: bool,
                 return_local_points: bool = False,
                 ):
        super().__init__()
        self.return_local_points = return_local_points
        self.no_heads = no_heads
        self.num_points = num_points
        self.is_multimer = is_multimer

        # Multimer requires this to be run with fp32 precision during training
        precision = torch.float32 if self.is_multimer else None
        self.linear = Linear(c_hidden, no_heads * 3 * num_points, precision=precision)

    def forward(self,
                activations: torch.Tensor,
                rigids: Union[Rigids, Rigid3Array],
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO: Needs to run in high precision during training
        points_local = self.linear(activations)
        out_shape = points_local.shape[:-1] + (self.no_heads, self.num_points, 3)

        if self.is_multimer:
            points_local = points_local.view(
                points_local.shape[:-1] + (self.no_heads, -1)
            )

        points_local = torch.split(
            points_local, points_local.shape[-1] // 3, dim=-1
        )

        points_local = torch.stack(points_local, dim=-1).view(out_shape)

        points_global = rigids[..., None, None].apply(points_local)

        if self.return_local_points:
            return points_global, points_local

        return points_global


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_qk_points: int,
        no_v_points: int,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(InvariantPointAttention, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc, bias=False)

        self.linear_q_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_k = Linear(self.c_s, hc, bias=False)
        self.linear_v = Linear(self.c_s, hc, bias=False)
        self.linear_k_points = PointProjection(
            self.c_s,
            self.no_qk_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_v_points = PointProjection(
            self.c_s,
            self.no_v_points,
            self.no_heads,
            is_multimer=True
        )

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-2)

    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid3Array,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_res, C_s] single representation
            z:
                [*, N_res, N_res, C_z] pair representation
            r:
                [*, N_res] transformation object
            mask:
                [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """

        a = 0.

        point_variance = (max(self.no_qk_points, 1) * 9.0 / 2)
        point_weights = math.sqrt(1.0 / point_variance)

        softplus = lambda x: torch.logaddexp(x, torch.zeros_like(x))

        head_weights = softplus(self.head_weights)
        point_weights = point_weights * head_weights

        #######################################
        # Generate scalar and point activations
        #######################################

        # [*, N_res, H, P_qk]
        q_pts = Vec3Array.from_array(self.linear_q_points(s, r))

        # [*, N_res, H, P_qk, 3]
        k_pts = Vec3Array.from_array(self.linear_k_points(s, r))

        pt_att = square_euclidean_distance(q_pts.unsqueeze(-3), k_pts.unsqueeze(-4), epsilon=0.)
        pt_att = torch.sum(pt_att * point_weights[..., None], dim=-1) * (-0.5)
        pt_att = pt_att.to(dtype=s.dtype)
        a = a + pt_att

        scalar_variance = max(self.c_hidden, 1) * 1.
        scalar_weights = math.sqrt(1.0 / scalar_variance)

        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        k = self.linear_k(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))

        q = q * scalar_weights
        a = a + torch.einsum('...qhc,...khc->...qkh', q, k)

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z)

        a = a + b

        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-1)
        a = a * math.sqrt(1. / 3)  # Normalize by number of logit terms (3)
        a = self.softmax(a)

        # [*, N_res, H * C_hidden]
        v = self.linear_v(s)

        # [*, N_res, H, C_hidden]
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        o = torch.einsum('...qkh, ...khc->...qhc', a, v)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, N_res, H, P_v, 3]
        v_pts = Vec3Array.from_array(self.linear_v_points(s, r))

        # [*, N_res, H, P_v]
        o_pt = v_pts[..., None, :, :, :] * a.unsqueeze(-1)
        o_pt = o_pt.sum(dim=-3)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(o_pt.shape[:-2] + (-1,))

        # [*, N_res, H, P_v]
        o_pt = r[..., None].apply_inverse_to_point(o_pt)
        o_pt_flat = [o_pt.x, o_pt.y, o_pt.z]
        o_pt_flat = [x.to(dtype=a.dtype) for x in o_pt_flat]

        # [*, N_res, H * P_v]
        o_pt_norm = o_pt.norm(epsilon=1e-8)

        o_pair = torch.einsum('...ijh, ...ijc->...ihc', a, z.to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *o_pt_flat, o_pt_norm, o_pair), dim=-1
            ).to(dtype=z.dtype)
        )

        return s

