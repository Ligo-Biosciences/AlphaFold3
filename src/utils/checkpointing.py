import importlib
from typing import Any, Tuple, List, Callable, Optional
import torch
import torch.utils.checkpoint
import functools

try:
    import deepspeed
    deepspeed_is_installed = True
except ImportError:
    deepspeed_is_installed = False


def get_checkpoint_fn():
    return deepspeed.checkpointing.checkpoint


def checkpoint_blocks(
    blocks: List[torch.nn.Module],
    args: Tuple[Any, ...],
    blocks_per_ckpt: Optional[int],
) -> Tuple[Any, ...]:
    """
    Chunk a list of blocks and run each chunk with activation checkpointing.
    Each block is a torch.nn.Module whose inputs are the outputs of the previous block.
    Checkpointing is only performed if training.

    Args:
        blocks: List of torch.nn.Module blocks.
        args: Tuple of arguments for the first block.
        blocks_per_ckpt: Number of blocks per checkpoint. If None, no checkpointing is performed.

    Returns:
        The output of the final block.
    """
    def execute_blocks(block_slice, inputs):
        for block in block_slice:
            inputs = block(*inputs)
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
        return inputs

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return execute_blocks(blocks, args)

    if blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn()
    
    for start in range(0, len(blocks), blocks_per_ckpt):
        end = start + blocks_per_ckpt
        args = checkpoint(
            lambda *inputs: execute_blocks(blocks[start:end], inputs),
            *args
        )
        args = args if isinstance(args, tuple) else (args,)

    return args
