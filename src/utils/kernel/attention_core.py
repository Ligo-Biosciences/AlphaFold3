import torch

from .triton.attention_core import attention_core_triton_kernel_wrapper


class FusedAttentionCoreFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask=None, bias=None):
        o = attention_core_triton_kernel_wrapper(q, k, v, mask, bias)
        return o


fused_attention_core = FusedAttentionCoreFunc.apply
