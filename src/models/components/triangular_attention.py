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

from functools import partialmethod, partial
from typing import Optional, List
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from src.models.components.primitives import Attention, LinearNoBias
from src.utils.chunk_utils import chunk_layer
from src.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(
            self, c_in, c_hidden, no_heads, starting=True, inf=1e9
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Per-head hidden channel dimension
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear = LinearNoBias(c_in, self.no_heads, init="normal")

        self.mha = Attention(
            c_q=self.c_in,
            c_k=self.c_in,
            c_v=self.c_in,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads
        )

    @torch.jit.ignore
    def _chunk(self,
               z: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int,
               use_deepspeed_evo_attention: bool = False,
               inplace_safe: bool = False,
               ) -> torch.Tensor:
        # triangle! triangle!
        mha_inputs = {
            "q_x": z,
            "kv_x": z,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
            _out=z if inplace_safe else None,
        )

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None,
                use_deepspeed_evo_attention: bool = False,
                inplace_safe: bool = False,
                ) -> torch.Tensor:
        """
        Args:
            z:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
            mask:
                [*, I, J] mask tensor
            chunk_size:
                The number of sub-batches per chunk. If multiple batch
                dimensions are specified, a "sub-batch" is defined as a single
                indexing of all batch dimensions simultaneously (s.timesteps. the
                number of sub-batches is the product of the batch dimensions).
            use_deepspeed_evo_attention:
                whether to use DeepSpeed's EvoFormer attention
            inplace_safe:
                in-place attention during inference and training

        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # [*, I, J]
            mask = z.new_ones(
                z.shape[:-1],
            )

        if not self.starting:
            z = z.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        z = self.layer_norm(z)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(z), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            z = self._chunk(
                z,
                biases,
                chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
            )
        else:
            z = self.mha(
                q_x=z,
                kv_x=z,
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            )

        if not self.starting:
            z = z.transpose(-2, -3)

        return z


class TriangleAttentionStartingNode(TriangleAttention):
    """
        Implements Algorithm 13.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
