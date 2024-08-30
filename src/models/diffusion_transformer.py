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

"""Diffusion transformer attention module."""
import torch
from torch import Tensor
import torch.nn as nn
from src.models.components.transition import ConditionedTransitionBlock
from src.models.components.attention_pair_bias import AttentionPairBias
from typing import Optional
from functools import partial
from src.utils.checkpointing import checkpoint_blocks
from typing import Tuple
from src.utils.tensor_utils import add


class DiffusionTransformerBlock(nn.Module):
    """Applies full self-attention and conditioned transition block."""

    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 16,
            no_blocks: int = 24,
            no_heads: int = 16,
            dropout: float = 0.0,
    ):
        """Initialize the DiffusionTransformerBlock module.
        Args:
            c_token:
                The number of channels for the token representation. Defaults to 384.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.
            no_blocks:
                Number of blocks.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
        """
        super().__init__()
        self.c_token = c_token
        self.c_pair = c_pair
        self.num_blocks = no_blocks
        self.num_heads = no_heads
        self.dropout = dropout

        self.attention_block = AttentionPairBias(
            c_token, c_pair, no_heads, dropout, input_gating=True, residual=False
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(c_token)

    def forward(
            self,
            single_repr: Tensor,  # (bs, S, n_tokens, c_token)
            single_proj: Tensor,  # (bs, 1, n_tokens, c_token)
            pair_repr: Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            mask: Optional[Tensor] = None,  # (bs, n_tokens)
            use_deepspeed_evo_attention: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the DiffusionTransformerBlock module. Algorithm 23 in AlphaFold3 supplement.
        TODO: the single_proj and pair_repr do not actually change as a result of this function.
            Returning them here is a bit misleading. Also, saving them between blocks is unnecessary.
        """
        single_repr = single_repr + self.attention_block(
            single_repr=single_repr,
            single_proj=single_proj,
            pair_repr=pair_repr,
            mask=mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )

        single_repr = add(
            single_repr,
            self.conditioned_transition_block(single_repr, single_proj),
            inplace=False
        )

        return single_repr, single_proj, pair_repr


class DiffusionTransformer(nn.Module):
    """DiffusionTransformer that applies multiple blocks of full self-attention and transition blocks."""

    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 16,
            no_blocks: int = 24,
            no_heads: int = 16,
            dropout=0.0,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
            compile_module: bool = False,
    ):
        """Initialize the DiffusionTransformer module.
        Args:
            c_token:
                The number of channels for the token representation. Defaults to 384.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.
            no_blocks:
                Number of blocks.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            blocks_per_ckpt:
                Number of blocks per checkpoint. If None, no checkpointing is used.
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            compile_module:
                Whether to compile the module.
        """
        super().__init__()
        self.c_token = c_token
        self.c_pair = c_pair
        self.num_blocks = no_blocks
        self.num_heads = no_heads
        self.dropout = dropout
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.compile_module = compile_module

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(c_token=c_token,
                                      c_pair=c_pair,
                                      no_blocks=no_blocks,
                                      no_heads=no_heads,
                                      dropout=dropout)
            for _ in range(no_blocks)
        ])

    def _prep_blocks(
            self,
            single_repr: Tensor,
            single_proj: Tensor,
            pair_repr: Tensor,
            mask: Optional[Tensor] = None,
            use_deepspeed_evo_attention: bool = True
    ):
        """Prepare the blocks for the forward pass."""
        blocks = [  # TODO: saving the pair_repr and single_proj between blocks is unnecessary
            partial(
                block if not self.compile_module else torch.compile(block),
                mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention
            )
            for block in self.blocks
        ]

        # Clear CUDA's GPU memory cache between blocks
        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]
        return blocks

    def forward(
            self,
            single_repr: Tensor,  # (*, S, N, c_s)
            single_proj: Tensor,  # (*, S, N, c_s)
            pair_repr: Tensor,  # (*, N, N, c_z)
            mask: Optional[Tensor] = None,  # (*, N)
            use_deepspeed_evo_attention: bool = True
    ):
        """Forward pass of the DiffusionTransformer module. Algorithm 23 in AlphaFold3 supplement.
        The DS4Science kernel for MSA row-wise attention is repurposed here for an efficient
        implementation of attention pair bias. The AttentionPairBias class is used in two
        main model components: the Pairformer and the Diffusion module. The main advantage of the
        kernel is in being able to accommodate a secondary batch-like dimension. In AlphaFold2, this
        is for N_seq in the MSA representation. In AlphaFold3, this is not needed in the Pairformer
        because we are using a single representation, so N_seq always equals 1. However, this is
        very useful in the diffusion module as multiple versions of the same input are created, and the
        same bias has to be added to this expanded representation throughout the DiffusionTransformer blocks.
        Here, we can use the N_seq dimension to host the samples per trunk which would make for a very memory
        efficient representation.
        Args:
            single_repr:
                [*, S, N, c_s] single representation, where S is the samples_per_trunk dimension.
            single_proj:
                [*, S, N, c_s] single projection
            pair_repr:
                [*, N, N, c_z] pair representation
            mask:
                [*, N] attention mask where 1.0 indicates valid token, 0.0 indicates invalid token.
            use_deepspeed_evo_attention:
                Whether to use deepspeed attention or not.
        """
        blocks = self._prep_blocks(
            single_repr=single_repr,
            single_proj=single_proj,
            pair_repr=pair_repr,
            mask=mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        # Run with grad checkpointing
        single_repr, single_proj, pair_repr = checkpoint_blocks(
            blocks,
            args=(single_repr, single_proj, pair_repr),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return single_repr
