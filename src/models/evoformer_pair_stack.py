import torch
from torch import nn
from collections import OrderedDict
from src.models.components.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from src.models.components.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing
)
from src.models.components.pair_transition import PairTransition
from src.models.components.dropout import DropoutRowwise, DropoutColumnwise


class EvoformerPairBlock(torch.nn.Module):
    """A block that evolves the pair representation."""

    def __init__(
            self,
            c_s: int = 384,
            n_heads: int = 4,
            c_hidden: int = 128,
            dropout_rate: float = 0.25,
    ):
        """Initialize the pair stack block.
        Args:
            c_s: the latent dimensionality of pair representation
            n_heads: number of attention heads in triangle attention
        """
        super().__init__()
        assert c_hidden % n_heads == 0, "c_hidden must be divisible by n_heads for multi-head attention."

        # Pair representation
        self.outgoing_dropout_rowwise = DropoutRowwise(dropout_rate)
        self.triangle_multiplication_outgoing = TriangleMultiplicationOutgoing(c_in=c_s, c_hidden=c_hidden)

        self.incoming_dropout_rowwise = DropoutRowwise(dropout_rate)
        self.triangle_multiplication_incoming = TriangleMultiplicationIncoming(c_in=c_s, c_hidden=c_hidden)

        self.starting_dropout_rowwise = DropoutRowwise(dropout_rate)
        self.triangle_attention_starting_node = TriangleAttentionStartingNode(c_in=c_s, c_hidden=c_hidden // n_heads,
                                                                              no_heads=n_heads)

        self.ending_dropout_columnwise = DropoutColumnwise(dropout_rate)
        self.triangle_attention_ending_node = TriangleAttentionEndingNode(c_in=c_s, c_hidden=c_hidden // n_heads,
                                                                          no_heads=n_heads)
        self.pair_transition = PairTransition(c_z=c_s, n=4)  # expand hidden dim by factor of 4

    def forward(self, z, mask=None):
        """Feedforward of pair stack block. Implements part of Alg. 6 in Jumper et al. 2021
        Args:
            z: the pair representation tensor [*, N_res, N_res, C_z]
            mask: input mask [*, N_res, N_res]
        """
        z += self.outgoing_dropout_rowwise(self.triangle_multiplication_outgoing(z, mask))
        z += self.incoming_dropout_rowwise(self.triangle_multiplication_incoming(z, mask))
        z += self.starting_dropout_rowwise(self.triangle_attention_starting_node(z, mask))
        z += self.ending_dropout_columnwise(self.triangle_attention_ending_node(z, mask))
        z += self.pair_transition(z, mask)
        return z


class EvoformerPairStack(torch.nn.Module):
    """A series of blocks that evolve the pair representation."""

    def __init__(
            self,
            n_blocks: int = 12,
            c_s: int = 384,
            n_heads: int = 4,
            c_hidden: int = 128,
            dropout_rate: float = 0.25
    ):
        super().__init__()
        self.blocks = nn.ModuleList([EvoformerPairBlock(c_s=c_s,
                                                        n_heads=n_heads,
                                                        c_hidden=c_hidden,
                                                        dropout_rate=dropout_rate)
                                     for _ in range(n_blocks)])

    def forward(self, z, mask=None):
        """Feedforward of Evoformer pair stack. Adapted from Alg. 6 in Jumper et al. 2021.
        :param z: the pair representation tensor [*, N_res, N_res, C_z]
        :param mask: input mask [*, N_res, N_res]
        """
        for block in self.blocks:
            z = block(z, mask)
        return z
