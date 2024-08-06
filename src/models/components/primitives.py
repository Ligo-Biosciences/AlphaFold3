# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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
import importlib
import math
from typing import Optional, Callable, List, Tuple, Sequence, Union
from functools import partialmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from src.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
ds4s_is_installed = deepspeed_is_installed and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
if deepspeed_is_installed:
    import deepspeed

if ds4s_is_installed:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

# from flash_attn.bert_padding import unpad_input
# from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
            precision=None
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")

        self.precision = precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.dtype
        deepspeed_is_initialized = (
                deepspeed_is_installed and
                deepspeed.comm.comm.is_initialized()
        )
        if self.precision is not None:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                return F.linear(x.to(dtype=self.precision),
                                self.weight.to(dtype=self.precision),
                                bias).to(dtype=d)

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return F.linear(x, self.weight.to(dtype=d), bias)

        return F.linear(x, self.weight, self.bias)


class LinearNoBias(Linear):
    """
        Convenience class for readability.
    """
    __init__ = partialmethod(Linear.__init__, bias=False)


class LayerNorm(nn.Module):
    # TODO: add elementwise_affine and bias option
    # TODO: do this with the Fastfold kernel (if the kernel is worth its salt)
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = (
                deepspeed_is_installed and
                deepspeed.comm.comm.is_initialized()
        )
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                out = F.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps
                )
        else:
            out = F.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out


class AdaLN(nn.Module):
    """Adaptive Layer Normalization."""

    def __init__(self, normalized_shape):
        super(AdaLN, self).__init__()
        # Layer norms
        self.a_layer_norm = nn.LayerNorm(normalized_shape,  # equivalent to scale=False, offset=False in Haiku
                                         elementwise_affine=False,
                                         bias=False)
        self.s_layer_norm = nn.LayerNorm(normalized_shape,  # equivalent to scale=True, offset=False in Haiku
                                         elementwise_affine=True,
                                         bias=False)

        # Linear layers for gating and the skip connection
        dim = normalized_shape if isinstance(normalized_shape, int) else normalized_shape[-1]
        self.to_gamma = nn.Sequential(
            Linear(dim, dim, init='gating'),
            nn.Sigmoid()
        )
        self.skip_linear = LinearNoBias(dim, dim, init='final')

    def forward(self, a, s):
        a = self.a_layer_norm(a)
        s = self.s_layer_norm(s)
        a = self.to_gamma(s) * a + self.skip_linear(s)
        return a


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    TODO: switch to fused softmax here.
    """
    d = t.dtype
    deepspeed_is_initialized = (
            deepspeed_is_installed and
            deepspeed.comm.comm.is_initialized()

    )
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.cuda.amp.autocast(enabled=False):
            s = F.softmax(t, dim=dim)
    else:
        s = F.softmax(t, dim=dim)
    return s


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        TODO: generalize this function to use the triton kernels
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # The qkv linear layers project no_heads * c_hidden and then split the dimensions
        split_heads = nn.Unflatten(dim=-1, unflattened_size=(self.no_heads, self.c_hidden))
        self.linear_q = nn.Sequential(
            LinearNoBias(self.c_q, self.c_hidden * self.no_heads, init="glorot"),
            split_heads
        )

        self.linear_k = nn.Sequential(
            LinearNoBias(self.c_k, self.c_hidden * self.no_heads, init="glorot"),
            split_heads
        )
        self.linear_v = nn.Sequential(
            LinearNoBias(self.c_v, self.c_hidden * self.no_heads, init="glorot"),
            split_heads
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.to_gamma = None
        if self.gating:
            self.to_gamma = nn.Sequential(
                Linear(self.c_q, self.c_hidden * self.no_heads, init="gating"),
                split_heads,
                nn.Sigmoid()
            )

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor,
                  apply_scale: bool = True
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H, C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K/V, H, C_hidden] -> [*, H, Q/K/V, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(
            self,
            o: torch.Tensor,
            q_x: torch.Tensor
    ) -> torch.Tensor:
        if self.to_gamma is not None:
            g = self.to_gamma(q_x)

            # [*, Q, H, C_hidden]
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
            use_deepspeed_evo_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
        Returns
            [*, Q, C_q] attention update
        """

        if biases is None:
            biases = []
        # DeepSpeed attention kernel applies scaling internally
        q, k, v = self._prep_qkv(q_x, kv_x,
                                 apply_scale=not use_deepspeed_evo_attention)

        if use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError(
                    "If use_deepspeed_evo_attention is True, you may only "
                    "provide up to two bias terms"
                )
            o = _deepspeed_evo_attn(q, k, v, biases)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


# @torch.jit.script
def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    """A stock PyTorch implementation of the attention mechanism.
    Args:
        query:
            [*, H, Q, C_hidden] query tensor
        key:
            [*, H, K/V, C_hidden] key tensor
        value:
            [*, H, K/V, C_value] value tensor
        biases:
            a list of biases that broadcast to [*, H, Q, K]
    Returns:
        the resultant tensor [*, H, Q, C_value]
    """

    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a = a + b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


@torch.jit.ignore
def _deepspeed_evo_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        biases: List[torch.Tensor],
):
    """""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k:
            [*, H, K, C_hidden] key data
        v:
            [*, H, V, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    """

    if not ds4s_is_installed:
        raise ValueError(
            "_deepspeed_evo_attn requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 1:
            raise AssertionError("Found no batch dimensions.")
        if no_batch_dims < 2:
            return x.reshape((x.shape[0],) + (1,) + x.shape[1:])
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # [*, H, Q/K, C_hidden] -> [*, Q/K, H, C_hidden]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
    # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
    # Cast to bf16 so kernel can be used during inference
    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                      k.to(dtype=torch.bfloat16),
                                      v.to(dtype=torch.bfloat16),
                                      [b.to(dtype=torch.bfloat16) for b in biases])
        o = o.to(dtype=orig_dtype)
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)

    o = o.reshape(orig_shape)
    return o
