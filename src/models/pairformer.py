"""The Pairformer stack fulfills a similar role to the Evoformer stack in AlphaFold2, the Pairformer stack uses just
a single representation rather than a representation for a subset of the MSA. The single representation does not
influence the pair representation, the pair representation is used to control information flow in the single
representation by biasing the Attention logits.
All transition blocks use SwiGlu."""

import torch
import torch.nn as nn

from typing import Tuple, Sequence, Optional
from functools import partial
from abc import ABC, abstractmethod

from src.models.components.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing
)
from src.models.components.transition import Transition
from src.models.components.dropout import (
    DropoutRowwise,
    DropoutColumnwise
)
from src.models.components.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from src.models.components.primitives import AttentionPairBias
from src.utils.tensor_utils import add
from src.utils.checkpointing import checkpoint_blocks
from src.utils.chunk_utils import ChunkSizeTuner, chunk_layer


class PairStack(nn.Module):
    def __init__(
            self,
            c_z: int,
            c_hidden_mul: int,
            c_hidden_pair_att: int,
            no_heads_pair: int,
            transition_n: int,
            pair_dropout: float,
            fuse_projection_weights: bool,
            inf: float,
            eps: float
    ):
        super(PairStack, self).__init__()

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_z,
                c_hidden_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_z,
                c_hidden_mul,
            )

        self.tri_att_start = TriangleAttentionStartingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = Transition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)
        self.ps_dropout_col_layer = DropoutColumnwise(pair_dropout)

    def forward(
            self,
            z: torch.Tensor,
            pair_mask: torch.Tensor,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
            _mask_trans: bool = True,
            _attn_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        pair_trans_mask = pair_mask if _mask_trans else None

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if not inplace_safe:
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update

        del tmu_update

        z = add(z,
                self.ps_dropout_row_layer(
                    self.tri_att_start(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = add(z,
                self.ps_dropout_col_layer(
                    self.tri_att_end(
                        z,
                        mask=pair_mask,
                        chunk_size=_attn_chunk_size,
                        use_memory_efficient_kernel=False,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    )
                ),
                inplace=inplace_safe,
                )

        z = add(z,
                self.pair_transition(
                    z, mask=pair_trans_mask, chunk_size=chunk_size,
                ),
                inplace=inplace_safe,
                )

        return z


class PairformerStackBlock(nn.Module):
    def __init__(
            self,
            c_s: int,
            c_z: int,
            c_hidden_mul: int = 128,
            c_hidden_pair_att: int = 32,
            no_heads_pair: int = 4,
            no_heads_single_att: int = 16,
            transition_n: int = 4,
            pair_dropout: float = 0.25,
            fuse_projection_weights: bool = False,
            inf: float = 1e8,
    ):
        super(PairformerStackBlock, self).__init__()
        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
        )

        self.attention = AttentionPairBias(
            dim=c_s,
            no_heads=no_heads_single_att,
            input_gating=False,  # no single representation gating within Pairformer
        )
        self.transition = Transition(
            c_s,
            transition_n,
        )

    def forward(
            self,
            s: torch.Tensor,  # (bs, n_tokens, c_s)
            z: torch.Tensor,  # (bs, n_tokens, c_z)
            single_mask: torch.Tensor,  # (bs, n_tokens)
            pair_mask: torch.Tensor,  # (bs, n_tokens, n_tokens)
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.pair_stack(
            z=z,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
        )
        s = self.attention(single_repr=s, pair_repr=z, mask=single_mask)
        s = self.transition(s)
        return s, z


class PairformerStack(nn.Module):
    def __init__(
            self,
            c_s: int,
            c_z: int,
            no_blocks: int = 48,
            c_hidden_mul: int = 128,
            c_hidden_pair_att: int = 32,
            no_heads_pair: int = 4,
            no_heads_single_att: int = 16,
            transition_n: int = 4,
            pair_dropout: float = 0.25,
            fuse_projection_weights: bool = False,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
            inf: float = 1e8,
    ):
        """
        Args:
            c_s:
                Single channel dimension
            c_z:
                Pair channel dimension
            no_blocks:
                Number of Evoformer blocks in the stack
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_pair:
                Number of heads in triangular attention
            no_heads_single_att:
                Number of heads in AttentionPairBias for the single representation
            transition_n:
                Expansion factor for the transition layers
            pair_dropout:
                Dropout rate for the pair stack
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            blocks_per_ckpt:
                Number of Pairformer blocks in each activation checkpoint
        """
        super(PairformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for i in range(no_blocks):
            block = PairformerStackBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                no_heads_single_att=no_heads_single_att,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
            )
            self.blocks.append(block)

    def _prep_blocks(
            self,
            s: torch.Tensor,  # (bs, n_tokens, c_s)
            z: torch.Tensor,  # (bs, n_tokens, c_z)
            single_mask: torch.Tensor,  # (bs, n_tokens)
            pair_mask: torch.Tensor,  # (bs, n_tokens, n_tokens)
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ):
        blocks = [
            partial(
                block,
                single_mask=single_mask,  # (bs, n_tokens)
                pair_mask=pair_mask,  # (bs, n_tokens, n_tokens)
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
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
            s: torch.Tensor,  # (bs, n_tokens, c_s)
            z: torch.Tensor,  # (bs, n_tokens, c_z)
            single_mask: torch.Tensor,  # (bs, n_tokens)
            pair_mask: torch.Tensor,  # (bs, n_tokens, n_tokens)
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, n_tokens, c_s] single representation
            z:
                [*, n_tokens, n_tokens, c_z] pair representation
            single_mask:
                [*, n_tokens] mask for the single representation
            pair_mask:
                [*, n_tokens, n_tokens] mask for the pair representation
            chunk_size:
                Inference-time sub-batch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            inplace_safe:
                Whether to use inference time inplace operations to save memory.
        """
        blocks = self._prep_blocks(
            s=s,  # We are very careful not to create references to these tensors in this function
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        return s, z

