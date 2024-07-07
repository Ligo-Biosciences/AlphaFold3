"""Relative position encoding for diffusion conditioning and the initial pair representation."""

import torch
from torch import nn
from src.models.components.primitives import Linear
from typing import Dict, Tuple
from src.utils.tensor_utils import one_hot


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

        # Compute total x dimensions for the linear projection
        input_dim = 2 * r_max + 2 + 2 * r_max + 2 + 2 * s_max + 2 + 1  # (relpos, rel_token, rel_chain, same_entity)
        self.linear_proj = Linear(input_dim, c_pair, bias=False)

    def forward(self, features: Dict[str, torch.Tensor], mask=None) -> torch.Tensor:
        """Computes relative position encoding. AlphaFold3 Supplement Algorithm 3.
        Args:
            features:
                input feature dictionary containing:
                    "residue_index":
                        [*, n_tokens] Residue number in the token's original x chain.
                    "token_index":
                        [*, n_tokens] Token number. Increases monotonically; does not restart at 1 for new chains
                    "asym_id":
                        [*, n_tokens] Unique integer for each distinct chain.
                    "entity_id":
                        [*, n_tokens] Unique integer for each distinct sequence.
                    "sym_id":
                        [*, n_tokens] Unique integer within chains of this sequence. e.g. if chains A, B and C
                        share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
            mask:
                [*, n_tokens] mask tensor
        Returns:
            [*, n_tokens, n_tokens, c_pair] relative position encoding tensor
        """

        # Same chain mask (bs, n_tokens, n_tokens)
        b_same_chain = torch.isclose(features["asym_id"][..., :, None],  features["asym_id"][..., None, :])

        # Same residue mask (bs, n_tokens, n_tokens)
        b_same_residue = torch.isclose(features["residue_index"][..., :, None],  features["residue_index"][..., None, :])

        b_same_entity = torch.isclose(features["entity_id"][..., :, None], features["entity_id"][..., None, :])
        b_same_entity = b_same_entity.unsqueeze(-1)  # (bs, n_tokens, n_tokens, 1)

        # Compute relative residue position encoding
        rel_pos = self.encode(features["residue_index"], b_same_chain, clamp_max=self.r_max)

        # Compute relative token position encoding
        rel_token = self.encode(features["token_index"], b_same_chain & b_same_residue, clamp_max=self.r_max)

        # Compute relative chain position encoding
        rel_chain = self.encode(features["asym_id"], b_same_chain, clamp_max=self.s_max)

        # Concatenate all features and project
        concat_features = torch.cat([rel_pos, rel_token, b_same_entity, rel_chain], dim=-1)
        p_ij = self.linear_proj(concat_features)

        # Mask the output
        if mask is not None:
            mask = (mask[..., :, None] * mask[..., None, :]).unsqueeze(-1)  # (bs, n_tokens, n_tokens, 1)
            p_ij = p_ij * mask
        return p_ij

    def encode(self,
               feature_tensor: torch.Tensor,
               condition_tensor: torch.Tensor,
               clamp_max: int) -> torch.Tensor:
        """Computes relative position encoding of an arbitrary tensor."""
        relative_dists = feature_tensor[..., None, :] - feature_tensor[..., :, None]
        d_ij = torch.where(
            condition_tensor,
            torch.clamp(torch.add(relative_dists, clamp_max), min=0, max=2 * clamp_max),
            torch.full_like(relative_dists, 2 * clamp_max + 1)
        )
        a_ij = one_hot(d_ij, v_bins=torch.arange(0, (2 * clamp_max + 2),
                                                 device=feature_tensor.device,
                                                 dtype=self.linear_proj.weight.dtype))
        return a_ij  # (bs, n_tokens, n_tokens, 2 * clamp_max + 2)
