import torch
from torch import nn
from src.models.components.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from src.models.components.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing
)
from src.models.components.outer_product_mean import OuterProductMean
from src.models.components.msa import (
    MSARowAttentionWithPairBias,
    MSATransition
)
from src.models.components.pair_transition import PairTransition
from src.models.components.dropout import DropoutRowwise, DropoutColumnwise


class EvoformerBlock(torch.nn.Module):
    """A block that evolves the pair and single representations."""

    def __init__(
            self,
            c_s: int = 384,
            c_z: int = 128,
            n_msa_row_attention_heads: int = 8,
            c_hidden_msa_per_head: int = 32,
            c_hidden_outer: int = 32,
            c_hidden_tri_mul: int = 128,
            n_triangle_attention_heads: int = 4,
            c_hidden_tri_att_per_head: int = 32,
            dropout_msa: float = 0.15,
            dropout_pair: float = 0.25,
    ):
        """Initialize the Evoformer block.
        Args:
            c_s:
                the latent dimensionality of the single representation
            c_z:
                the latent dimensionality of the pair representation
            n_msa_row_attention_heads:
                number of MSA/single representation row-wise attention heads
            c_hidden_msa_per_head:
                number of hidden dimensions per head in MSARowAttention
            c_hidden_outer:
                number of hidden channels in the outer product mean module
            c_hidden_tri_mul:
                number of hidden channels in triangular multiplicative update
            n_triangle_attention_heads:
                number of attention heads in MSA
            c_hidden_tri_att_per_head:
                number of hidden channels in triangular attention per head
            dropout_msa:
                dropout rate in MSA/single representation
            dropout_pair:
                dropout rate in the column/row-wise pair representation
        """
        super().__init__()

        # MSA/Single Representation
        self.msa_row_attention = MSARowAttentionWithPairBias(c_m=c_s,
                                                             c_z=c_z,
                                                             c_hidden=c_hidden_msa_per_head,
                                                             no_heads=n_msa_row_attention_heads)
        self.outer_product_mean = OuterProductMean(c_m=c_s, c_z=c_z, c_hidden=c_hidden_outer)
        self.msa_transition = MSATransition(c_m=c_s, n=4)  # expand hidden c_hidden by factor of 4
        self.msa_dropout = DropoutRowwise(dropout_msa)

        # Pair representation
        self.outgoing_dropout_rowwise = DropoutRowwise(dropout_pair)
        self.triangle_multiplication_outgoing = TriangleMultiplicationOutgoing(c_in=c_z,
                                                                               c_hidden=c_hidden_tri_mul)

        self.incoming_dropout_rowwise = DropoutRowwise(dropout_pair)
        self.triangle_multiplication_incoming = TriangleMultiplicationIncoming(c_in=c_z,
                                                                               c_hidden=c_hidden_tri_mul)

        self.starting_dropout_rowwise = DropoutRowwise(dropout_pair)
        self.triangle_attention_starting_node = TriangleAttentionStartingNode(c_in=c_z,
                                                                              c_hidden=c_hidden_tri_att_per_head,
                                                                              no_heads=n_triangle_attention_heads)

        self.ending_dropout_columnwise = DropoutColumnwise(dropout_pair)
        self.triangle_attention_ending_node = TriangleAttentionEndingNode(c_in=c_z,
                                                                          c_hidden=c_hidden_tri_att_per_head,
                                                                          no_heads=n_triangle_attention_heads)
        self.pair_transition = PairTransition(c_z=c_z, n=4)  # expand hidden c_hidden by factor of 4

    def forward(self, m, z, mask=None):
        """Feedforward of pair stack block. Implements part of Alg. 6 in Jumper et al. 2021
        Args:
            m: the single representation tensor [*, 1, N_res, C_z] (the first element is size 1 for compatibility)
            z: the pair representation tensor [*, N_res, N_res, C_z]
            mask: Optional [*, N_res] sequence mask
        TODO: when exactly is the mask required here? I understand why the loss should be masked but not
         the internal operations of the Evoformer.
        """
        # Mask shape wrangling
        msa_mask = None
        pair_mask = None
        if mask is not None:
            msa_mask = mask[:, None, :]  # [*, 1, N_res] single representation mask
            pair_mask = mask[:, :, None] * mask[:, None, :]  # [*, N_res, N_res] input mask

        # MSA Stack
        m = m + self.msa_dropout(self.msa_row_attention(m, z, msa_mask))
        m = m + self.msa_transition(m, msa_mask)

        # Communication
        z = z + self.outer_product_mean(m, msa_mask)

        # Pair Stack
        z = z + self.outgoing_dropout_rowwise(self.triangle_multiplication_outgoing(z, pair_mask))
        z = z + self.incoming_dropout_rowwise(self.triangle_multiplication_incoming(z, pair_mask))
        z = z + self.starting_dropout_rowwise(self.triangle_attention_starting_node(z, pair_mask))
        z = z + self.ending_dropout_columnwise(self.triangle_attention_ending_node(z, pair_mask))
        z = z + self.pair_transition(z, pair_mask)
        return m, z


class EvoformerStack(torch.nn.Module):
    """A series of blocks that evolve the single and the pair representation."""

    def __init__(
            self,
            n_blocks: int = 12,
            c_s: int = 384,
            c_z: int = 128,
            n_msa_row_attention_heads: int = 8,
            c_hidden_msa_per_head: int = 32,
            c_hidden_outer: int = 32,
            c_hidden_tri_mul: int = 128,
            n_triangle_attention_heads: int = 4,
            c_hidden_tri_att_per_head: int = 32,
            dropout_msa: float = 0.15,
            dropout_pair: float = 0.25,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([EvoformerBlock(c_s=c_s,
                                                    c_z=c_z,
                                                    n_msa_row_attention_heads=n_msa_row_attention_heads,
                                                    c_hidden_msa_per_head=c_hidden_msa_per_head,
                                                    c_hidden_outer=c_hidden_outer,
                                                    c_hidden_tri_mul=c_hidden_tri_mul,
                                                    n_triangle_attention_heads=n_triangle_attention_heads,
                                                    c_hidden_tri_att_per_head=c_hidden_tri_att_per_head,
                                                    dropout_msa=dropout_msa,
                                                    dropout_pair=dropout_pair)
                                     for _ in range(n_blocks)])

    def forward(self, s, z, mask=None):
        """Feedforward of Evoformer pair stack. Adapted from Alg. 6 in Jumper et al. 2021.
        Args:
            s:
                the single representation tensor [*, N_res, C_s]
            z:
                the pair representation tensor [*, N_res, N_res, C_z]
            mask:
                Optional [*, N_res] sequence mask
        """
        # Making the m tensor consistent with [*, N_seq, N_res, C_s] format
        m = s[:, None, :, :]  # [*, 1, N_res, C_s]

        for block in self.blocks:
            m, z = block(m, z, mask)

        evoformer_output_dict = {
            "single": m.squeeze(-3),  # remove the singleton dimension
            "pair": z
        }
        return evoformer_output_dict
