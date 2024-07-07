import torch
from torch import Tensor
from torch import nn
from typing import Optional
from functools import partial
from src.models.pairformer import PairStack
from src.utils.checkpointing import checkpoint_blocks


class TemplatePairStack(nn.Module):
    """Implements the PairStack that processes the templates.
    Note: DeepMind writes 'PairformerStack' in the pseudocode, but PairformerStack also requires a single
    representation. We assume this is meant to be the PairStack similar to the one used in AlphaFold2, so we
    implement the TemplatePairStack here."""

    def __init__(
            self,
            no_blocks: int = 2,
            c_template: int = 32,
            clear_cache_between_blocks: bool = False,
            blocks_per_ckpt: int = 1
    ):
        super(TemplatePairStack, self).__init__()
        self.blocks = nn.ModuleList([PairStack(c_z=c_template) for _ in range(no_blocks)])
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks_per_ckpt = blocks_per_ckpt

    def _prep_blocks(
            self,
            z: Tensor,
            pair_mask: Tensor,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False
    ):
        blocks = [
            partial(
                block,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe
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
            z: Tensor,
            pair_mask: Tensor,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False
    ) -> Tensor:

        blocks = self._prep_blocks(
            z=z,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        (z,) = checkpoint_blocks(
            blocks,
            args=(z,),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return z
