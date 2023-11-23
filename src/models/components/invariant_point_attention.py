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


from functools import reduce
import math
import sys
from operator import mul
import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence

from src.utils.precision_utils import is_fp16_enabled
from src.utils.rigid_utils import Rotations, Rigids

from src.models.components.primitives import Linear, ipa_point_weights_init_
from src.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


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

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        hpv = self.no_heads * self.no_v_points * 3

        self.linear_b = Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
                self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigids,
            mask: torch.Tensor,
            inplace_safe: bool = False,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
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
        if _offload_inference and inplace_safe:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if _offload_inference:
            assert (sys.getrefcount(z[0]) == 2)
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                a = torch.matmul(
                    permute_final_dims(q.float(), (1, 0, 2)),  # [*, H, N_res, C_hidden]
                    permute_final_dims(k.float(), (1, 2, 0)),  # [*, H, C_hidden, N_res]
                )
        else:
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
            )

        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        if inplace_safe:
            pt_att *= pt_att
        else:
            pt_att = pt_att ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        if inplace_safe:
            pt_att *= head_weights
        else:
            pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        if inplace_safe:
            a += pt_att
            del pt_att
            a += square_mask.unsqueeze(-3)
            # in-place softmax
            attn_core_inplace_cuda.forward_(
                a,
                reduce(mul, a.shape[:-1]),
                a.shape[-1],
            )
        else:
            a = a + pt_att
            a = a + square_mask.unsqueeze(-3)
            a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v]
        if inplace_safe:
            v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
            o_pt = [
                torch.matmul(a, v.to(a.dtype))
                for v in torch.unbind(v_pts, dim=-3)
            ]
            o_pt = torch.stack(o_pt, dim=-3)
        else:
            o_pt = torch.sum(
                (
                        a[..., None, :, :, None]
                        * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
                ),
                dim=-2,
            )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_res, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            ).to(dtype=z[0].dtype)
        )

        return s
