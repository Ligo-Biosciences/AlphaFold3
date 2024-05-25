""""""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components.primitives import AdaLN


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
        self.hidden_gating_linear = nn.Linear(input_dim, n * input_dim, bias=False)
        self.hidden_linear = nn.Linear(input_dim, n * input_dim, bias=False)
        self.output_linear = nn.Linear(input_dim * n, input_dim)
        self.output_gating_linear = nn.Linear(input_dim, input_dim)
        self.output_gating_linear.bias = nn.Parameter(torch.ones(input_dim) * -2.0)

    def forward(self, a, s):
        a = self.ada_ln(a, s)
        b = F.silu(self.hidden_gating_linear(a)) * self.hidden_linear(a)
        # Output projection (from adaLN-Zero)
        a = F.sigmoid(self.output_gating_linear(s)) * self.output_linear(b)
        return a
