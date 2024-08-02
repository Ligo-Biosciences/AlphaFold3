from functools import reduce
from operator import mul
import torch

from .triton.softmax import softmax_triton_kernel_wrapper
from .triton.softmax import softmax_grad_triton_kernel_wrapper


class FusedSoftmaxFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask=None, bias=None):
        input_ = input.contiguous()
        mask_, bias_ = None, None
        ctx.use_bias = False
        if mask is not None:
            mask_ = mask.contiguous()
        if bias is not None:
            bias_ = bias.contiguous()
            ctx.use_bias = True
        ctx.cols = input_.shape[-1]
        ctx.rows = reduce(mul, input.shape[:-1])
        output = softmax_triton_kernel_wrapper(input_, mask_, bias_, ctx.rows, ctx.cols)
        ctx.save_for_backward(output, mask_)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        output, mask_ = ctx.saved_tensors
        grad_input = softmax_grad_triton_kernel_wrapper(grad_output, output, ctx.rows, ctx.cols)
        grad_bias = None
        if ctx.use_bias:
            grad_bias = torch.sum(grad_input, dim=1, keepdim=True)

        return grad_input, None, grad_bias


fused_softmax = FusedSoftmaxFunc.apply
