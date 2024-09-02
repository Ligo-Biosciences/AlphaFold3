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

"""Utilities for refactoring the block prep and checkpointing logic."""
import torch
from functools import partial
from typing import Optional, List, Callable, Tuple, Any
from torch import Tensor
from src.utils.checkpointing import checkpoint_blocks


def prep_blocks(
        blocks: List[Callable], 
        clear_cache_between_blocks: bool, 
        **kwargs: Any
) -> List[Callable]:
    """Prepare the blocks for the forward pass."""
    prepared_blocks = [
        partial(block, **kwargs)
        for block in blocks
    ]

    # Clear CUDA's GPU memory cache between blocks
    if clear_cache_between_blocks:
        def block_with_cache_clear(block, *args, **kwargs):
            torch.cuda.empty_cache()
            return block(*args, **kwargs)

        prepared_blocks = [partial(block_with_cache_clear, b) for b in prepared_blocks]
    return prepared_blocks


def forward_with_checkpointing(
        blocks: List[Callable], 
        args: Tuple[Tensor, ...], 
        blocks_per_ckpt: Optional[int]
) -> Tuple[Tensor, ...]:
    """Run the blocks with gradient checkpointing."""
    if not torch.is_grad_enabled():
        blocks_per_ckpt = None

    # Run with grad checkpointing
    return checkpoint_blocks(
        blocks,
        args=args,
        blocks_per_ckpt=blocks_per_ckpt,
    )