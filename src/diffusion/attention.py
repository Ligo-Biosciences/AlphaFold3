"""Diffusion transformer attention module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.transition import ConditionedTransitionBlock
from src.models.components.primitives import AttentionPairBias


class DiffusionTransformer(nn.Module):
    """DiffusionTransformer that applies multiple blocks of full self-attention and transition blocks."""
    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 16,
            num_blocks: int = 24,
            num_heads: int = 16,
            dropout=0.0,
            device=None,
            dtype=None,
    ):
        """Initialize the DiffusionTransformer module.
        Args:
            c_token:
                The number of channels for the token representation. Defaults to 384.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.
            num_blocks:
                Number of blocks.
            num_heads:
                Number of parallel attention heads. Note that embed_dim will be split across num_heads
                (i.e. each head will have dimension embed_dim // num_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
        """
        super().__init__()
        self.c_token = c_token
        self.c_pair = c_pair
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        self.attention_blocks = nn.ModuleList(
            [AttentionPairBias(c_token, c_pair, num_heads, dropout, device, dtype)
             for _ in range(num_blocks)]
        )
        self.conditioned_transition_blocks = nn.ModuleList(
            [ConditionedTransitionBlock(c_token) for _ in range(num_blocks)]
        )

    def forward(self, single_repr, single_proj, pair_repr, mask=None):
        """Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement."""
        for i in range(self.num_blocks):
            b = self.attention_blocks[i](single_repr, single_proj, pair_repr, mask)
            single_repr = b + self.conditioned_transition_blocks[i](single_repr, single_proj)
        return single_repr
