"""The MSA Module in AlphaFold3 fulfills a similar role to the Extra MSA Stack in AlphaFold2 and hence a fairly
similar architecture to AlphaFold-Multimer in the block. It samples a new i.i.d. random subset of the MSA for each
recycling iteration, the MSA sequences and input features then get embedded into representation m_si for each token n
each sequence in the MSA.

The overall structure of the block is very similar to the Pairformer Stack, where the MSA representation fulfills a
role similar to the single representation. The individual blocks here are similar to the extra MSA stack in AlphaFold
2, the difference here is that the attention is independently performed for each row of the MSA and that attention
weights are entirely projected from the pair representation, i.e. there is no key-query based attention. This also
means that each row of the MSA combines information via attention in the same way, this reduces computation and
memory usage in the attention. The MSA attention layer employs the same gating mechanism as the other attention
layers. Otherwise, this part of the model works the same as AlphaFold 2, meaning the pair representation gets passed
through Triangular Multiplicative Update and Triangular self-attention layers and a transition block. In all the
transition blocks we use SwiGLU instead of ReLU. Conceptually a difference to AlphaFold 2 is that here we do not
combine information across different rows of the MSA directly, but rather all information has to flow via the pair
representation. The motivation behind this is that the pair representation should contain as much information as
possible about the proteins or nucleic acids as it forms the backbone for the rest of the network. """

import torch
from torch import Tensor
from torch import nn
from typing import Optional
from src.models.components.primitives import LinearNoBias, LayerNorm
from src.models.pairformer import PairStack


class MSAPairWeightedAveraging(nn.Module):
    def __init__(
            self,
            c_msa: int,
            c_z: int,
            c_hidden: int,
            no_heads: int,
            inf: float = 1e8
    ):
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_msa = c_msa
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        # MSA
        self.msa_ln = LayerNorm(c_msa)
        self.msa_proj = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='glorot'),
            nn.Unflatten(dim=-1, unflattened_size=(no_heads, c_hidden))  # split the heads
        )
        self.to_gamma = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='gating'),
            nn.Unflatten(dim=-1, unflattened_size=(no_heads, c_hidden)),  # split the heads
            nn.Sigmoid()
        )

        # Pair
        self.proj_pair_bias = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_hidden, no_heads)
        )

        # Output projection
        self.output_proj = LinearNoBias(no_heads * c_hidden, c_msa, init='final')

        self.softmax = nn.Softmax(dim=-2)

    def forward(
            self,
            m: Tensor,
            z: Tensor,
            msa_mask: Optional[Tensor] = None,
            z_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embeddings
            z:
                [*, N_res, N_res, C_z] pair embeddings
            msa_mask:
                [*, N_seq, N_res] MSA mask
            z_mask:
                [*, N_res, N_res] pair mask
        """
        *_, n_seq, n_res, _ = m.shape

        # Input projections
        m_ln = self.msa_ln(m)
        v = self.msa_proj(m_ln)  # (*, seq, res, heads, c_hidden)
        b = self.proj_pair_bias(z)  # (*, res, res, no_heads)
        g = self.to_gamma(m_ln)  # (*, seq, res, heads, c_hidden)

        del m_ln

        # Masking and shape wrangling
        if z_mask is not None:
            z_mask = z_mask.unsqueeze(-1)  # (*, N_res, N_res, 1)
            z_mask = self.inf * (z_mask - 1)  # mask before softmax
            b = b + z_mask

        if msa_mask is not None:
            v = v * msa_mask.unsqueeze(-1).unsqueeze(-1)
        new_v_shape = (v.shape[:-3] + (n_seq, n_res, n_res, self.no_heads, self.c_hidden))
        v = v.unsqueeze(-3).expand(new_v_shape)  # (*, seq, res, res, heads, c_hidden)

        # Weighted average with gating
        weights = self.softmax(b)
        weights = weights.unsqueeze(-4).unsqueeze(-1)  # (*, 1, res, res, heads, 1)
        o = g * torch.sum(v * weights, dim=-3)  # (*, seq, res, heads, c_hidden)

        # Output projection
        output = self.output_proj(
            o.reshape((m.shape[:-1], self.c_hidden * self.no_heads))  # (*, seq, res, c_hidden * heads)
        )
        return output


class MSAModule(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        # SampleRandomWithoutReplacement
        # Embed the MSA
        # for all blocks:
        #  Communication
        #  z_ij = z_ij + OuterProductMean(m_si)
        #  MSA stack
        #  z_ij = z_ij + MSAPairWeightedAveraging(m_si, z_ij)
        #  Pair stack
        #  z_ij = PairStack(z_ij)
        #  Transition
        #  z_ij = Transition(z_ij)
        pass
