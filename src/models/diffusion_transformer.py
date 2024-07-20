"""Diffusion transformer attention module."""
import torch
from torch import Tensor
import torch.nn as nn
from src.models.components.transition import ConditionedTransitionBlock
from src.models.components.primitives import AttentionPairBias
from typing import Optional
from functools import partial
from src.utils.checkpointing import checkpoint_blocks
from typing import Tuple


class DiffusionTransformerBlock(nn.Module):
    """Applies full self-attention and conditioned transition block."""

    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 16,
            num_blocks: int = 24,
            no_heads: int = 16,
            dropout: float = 0.0,
    ):
        """Initialize the DiffusionTransformerBlock module.
        Args:
            c_token:
                The number of channels for the token representation. Defaults to 384.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.
            num_blocks:
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
        self.num_blocks = num_blocks
        self.num_heads = no_heads
        self.dropout = dropout

        self.attention_block = AttentionPairBias(c_token, c_pair, no_heads, dropout, input_gating=True)
        self.conditioned_transition_block = ConditionedTransitionBlock(c_token)

    def forward(
            self,
            single_repr: Tensor,  # (bs, n_tokens, c_token)
            single_proj: Tensor,  # (bs, n_tokens, c_token)
            pair_repr: Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            mask: Optional[Tensor] = None,
            use_flash: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the DiffusionTransformerBlock module. Algorithm 23 in AlphaFold3 supplement.
        TODO: the single_proj and pair_repr do not actually change as a result of this function.
            Returning them here is a bit misleading. Also, saving them between blocks is unnecessary.
        """
        b = self.attention_block(
            single_repr=single_repr,
            single_proj=single_proj,
            pair_repr=pair_repr,
            mask=mask,
            use_flash=use_flash
        )
        single_repr = b + self.conditioned_transition_block(single_repr, single_proj)
        return single_repr, single_proj, pair_repr


class DiffusionTransformer(nn.Module):
    """DiffusionTransformer that applies multiple blocks of full self-attention and transition blocks."""

    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 16,
            num_blocks: int = 24,
            num_heads: int = 16,
            dropout=0.0,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
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
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            blocks_per_ckpt:
                Number of blocks per checkpoint. If None, no checkpointing is used.
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super().__init__()
        self.c_token = c_token
        self.c_pair = c_pair
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(c_token=c_token,
                                      c_pair=c_pair,
                                      num_blocks=num_blocks,
                                      no_heads=num_heads,
                                      dropout=dropout)
            for _ in range(num_blocks)
        ])

    def _prep_blocks(
            self,
            single_repr: Tensor,
            single_proj: Tensor,
            pair_repr: Tensor,
            mask: Optional[Tensor] = None,
            use_flash: bool = True
    ):
        """Prepare the blocks for the forward pass."""
        blocks = [  # TODO: saving the pair_repr and single_proj between blocks is unnecessary
            partial(
                block,
                mask=mask,
                use_flash=use_flash
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
            single_repr: Tensor,
            single_proj: Tensor,
            pair_repr: Tensor,
            mask: Optional[Tensor] = None,
            use_flash: bool = True
    ):
        """Forward pass of the DiffusionTransformer module. Algorithm 23 in AlphaFold3 supplement."""
        blocks = self._prep_blocks(
            single_repr=single_repr,
            single_proj=single_proj,
            pair_repr=pair_repr,
            mask=mask,
            use_flash=use_flash
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
