"""Diffusion conditioning."""

import torch
from torch import nn
import math
from src.models.components.primitives import Linear
from typing import Dict
from torch.nn import functional as F


class FourierEmbedding(nn.Module):
    """Fourier embedding for diffusion conditioning."""
    def __init__(self, embed_dim):
        super(FourierEmbedding, self).__init__()
        self.embed_dim = embed_dim
        # Randomly generate weight/bias once before training
        self.weight = nn.Parameter(torch.randn((embed_dim,)))
        self.bias = nn.Parameter(torch.randn((embed_dim,)))

    def forward(self, t):
        """Compute embeddings"""
        return torch.cos(torch.tensor(2 * math.pi) * (t * self.weight + self.bias))


class RelativePositionEncoding(nn.Module):
    """Relative position encoding for diffusion conditioning."""
    def __init__(
            self,
            c_pair: int,
            r_max: int = 32,
            s_max: int = 2
    ):
        """Initializes the relative position encoding.
        Args:
            c_pair:
                dimensions of the pair representation
            r_max:
                maximum residue distance, plus or minus r_max
            s_max:
                maximum asym_id distance, plus or minus s_max
        """
        super(RelativePositionEncoding, self).__init__()
        self.c_pair = c_pair
        self.r_max = r_max
        self.s_max = s_max

        # Compute total input dimensions for the linear projection
        input_dim = 2 * r_max + 2 + 2 * r_max + 2 + 2 * s_max + 2 + 1  # (relpos, rel_token, rel_chain, same_entity)
        self.linear_proj = Linear(input_dim, c_pair, bias=False)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes relative position encoding. AlphaFold3 Supplement Algorithm 3.
        Args:
            features:
                input feature dictionary containing:
                    "residue_index":
                        [*, n_tokens] Residue number in the token's original input chain.
                    "token_index":
                        [*, n_tokens] Token number. Increases monotonically; does not restart at 1 for new chains
                    "asym_id":
                        [*, n_tokens] Unique integer for each distinct chain.
                    "entity_id":
                        [*, n_tokens] Unique integer for each distinct sequence.
                    "sym_id":
                        [*, n_tokens] Unique integer within chains of this sequence. e.g. if chains A, B and C
                        share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
        Returns:
            relative position encoding tensor
        """
        # Same chain mask (bs, n_tokens, n_tokens)
        b_same_chain = features["asym_id"][:, :, None] == features["asym_id"][:, None, :]  # (bs, n_tokens, n_tokens)

        # Same residue mask (bs, n_tokens, n_tokens)
        b_same_residue = features["residue_index"][:, :, None] == features["residue_index"][:, None, :]

        b_same_entity = features["entity_id"][:, :, None] == features["entity_id"][:, None, :]
        b_same_entity = b_same_entity.unsqueeze(-1)  # (bs, n_tokens, n_tokens, 1)

        # Compute relative residue position encoding
        rel_pos = RelativePositionEncoding.encode(features["residue_index"], b_same_chain, clamp_max=self.r_max)

        # Compute relative token position encoding
        rel_token = RelativePositionEncoding.encode(features["token_index"], b_same_chain & b_same_residue,
                                                    clamp_max=self.r_max)

        # Compute relative chain position encoding
        rel_chain = RelativePositionEncoding.encode(features["asym_id"], b_same_chain, clamp_max=self.s_max)

        p_ij = self.linear_proj(torch.cat([rel_pos, rel_token, b_same_entity, rel_chain], dim=-1).float())
        return p_ij

    @staticmethod
    def encode(feature_tensor: torch.Tensor,
               condition_tensor: torch.Tensor,
               clamp_max: int) -> torch.Tensor:
        """Computes relative position encoding of an arbitrary tensor."""
        relative_dists = feature_tensor[:, None, :] - feature_tensor[:, :, None]
        d_ij = torch.where(
            condition_tensor,
            torch.clamp(torch.add(relative_dists, clamp_max), min=0, max=2*clamp_max),
            torch.full_like(relative_dists, 2*clamp_max + 1)
        )
        return F.one_hot(d_ij, num_classes=2 * clamp_max + 2)  # (bs, n_tokens, n_tokens, 2 * clamp_max + 2)


