"""Transition blocks in AlphaFold3"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from src.models.components.primitives import AdaLN
from src.models.components.primitives import Linear, LinearNoBias


class Transition(nn.Module):
    """A transition block for a residual update.
    Warning: at initialization, the final output linear layer is initialized with zeros."""
    def __init__(self, input_dim: int, n: int = 4):
        """
        Args:
            input_dim:
                Channels of the input tensor
            n:
                channel expansion factor for hidden dimensions
        """
        super(Transition, self).__init__()
        self.layer_norm = LayerNorm(input_dim)
        self.linear_1 = LinearNoBias(input_dim, n * input_dim, init='relu')
        self.linear_2 = LinearNoBias(input_dim, n * input_dim, init='default')
        self.output_linear = LinearNoBias(input_dim * n, input_dim, init='final')

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.silu(self.linear_1(x)) * self.linear_2(x)
        return self.output_linear(x)


class ConditionedTransitionBlock(nn.Module):
    """SwiGLU transition block with adaptive layer norm."""
    def __init__(self,
                 input_dim: int,
                 n: int = 2):
        """
        Args:
            input_dim:
                Channels of the input tensor
            n:
                channel expansion factor for hidden dimensions
        """
        super(ConditionedTransitionBlock, self).__init__()
        self.ada_ln = AdaLN(input_dim)
        self.hidden_gating_linear = LinearNoBias(input_dim, n * input_dim, init='relu')
        self.hidden_linear = LinearNoBias(input_dim, n * input_dim, init='default')
        self.output_linear = Linear(input_dim * n, input_dim, init='default')
        # TODO: check if this is in line with the adaLN-Zero initialization
        self.output_gating_linear = Linear(input_dim, input_dim, init='gating')
        self.output_gating_linear.bias = nn.Parameter(torch.ones(input_dim) * -2.0)  # gate values will be ~0.11

    def forward(self, a, s):
        a = self.ada_ln(a, s)
        b = F.silu(self.hidden_gating_linear(a)) * self.hidden_linear(a)
        # Output projection (from adaLN-Zero)
        a = F.sigmoid(self.output_gating_linear(s)) * self.output_linear(b)
        return a
