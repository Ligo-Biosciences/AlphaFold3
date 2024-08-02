# part of code modified from https://github.com/NVIDIA/apex
import numbers

import torch
from torch.nn.parameter import Parameter

from .triton.layer_norm import LayerNormTritonFunc


class FusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):
        super(FusedLayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        if len(input.shape) >= 3 and input.shape[-3] > 4000:
            out = torch.empty_like(input)
            # set max chunk_size = dim / 2, to max compute efficiency
            chunk_size = min(4000 * 4000 // input.shape[-3], (input.shape[-3] + 1) // 2)
            if len(input.shape) == 3:
                for i in range(input.shape[-3]):
                    out[i:i + chunk_size] = self.kernel_forward(input[i:i + chunk_size])
            elif len(input.shape) == 4:
                for j in range(input.shape[-4]):
                    for i in range(0, input.shape[-3], chunk_size):
                        out[j, i:i + chunk_size] = self.kernel_forward(input[j, i:i + chunk_size])
            else:
                raise RuntimeError("Shape" + input.shape + "not implemented for layernorm yet!")
            return out
        else:
            return self.kernel_forward(input)

    def kernel_forward(self, input):
        return LayerNormTritonFunc.apply(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )
