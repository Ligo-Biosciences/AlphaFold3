"""Diffusion conditioning."""

import torch
from torch import nn
from src.models.components.primitives import Linear
from src.models.components.relative_position_encoding import RelativePositionEncoding
from src.models.components.transition import Transition
from typing import Dict, Tuple
from src.utils.checkpointing import get_checkpoint_fn
checkpoint = get_checkpoint_fn()


class FourierEmbedding(nn.Module):
    """Fourier embedding for diffusion conditioning."""
    def __init__(self, embed_dim):
        super(FourierEmbedding, self).__init__()
        self.embed_dim = embed_dim
        # Randomly generate weight/bias once before training
        self.weight = nn.Parameter(torch.randn((1, embed_dim)))
        self.bias = nn.Parameter(torch.randn((1, embed_dim)))

    def forward(self, t):
        """Compute embeddings"""
        two_pi = torch.tensor(2 * 3.1415, device=t.device, dtype=t.dtype)
        return torch.cos(two_pi * (t * self.weight + self.bias))


class DiffusionConditioning(nn.Module):
    """Diffusion conditioning module."""

    def __init__(
            self,
            c_token: int = 384,
            c_pair: int = 128,
    ):
        """Initializes the diffusion conditioning module.
        Args:
            c_token:
                dimensions of the token representation
            c_pair:
                dimensions of the token pair representation
        """
        super(DiffusionConditioning, self).__init__()
        self.c_token = c_token
        self.c_pair = c_pair

        # Pair conditioning
        self.relative_position_encoding = RelativePositionEncoding(c_pair)
        self.pair_layer_norm = nn.LayerNorm(2 * c_pair)  # z_trunk + relative_position_encoding
        self.linear_pair = Linear(2 * c_pair, c_pair, bias=False)
        self.pair_transitions = nn.ModuleList([Transition(input_dim=c_pair, n=2) for _ in range(2)])

        # Single conditioning
        self.single_layer_norm = nn.LayerNorm(2 * c_token)  # s_trunk + s_inputs
        self.linear_single = Linear(2 * c_token, c_token, bias=False)
        self.fourier_embedding = FourierEmbedding(embed_dim=256)  # 256 is the default value in the paper
        self.fourier_layer_norm = nn.LayerNorm(256)
        self.linear_fourier = Linear(256, c_token, bias=False)
        self.single_transitions = nn.ModuleList([Transition(input_dim=c_token, n=2) for _ in range(2)])

    def _forward(
            self,
            timesteps: torch.Tensor,  # timestep (bs, 1)
            features: Dict[str, torch.Tensor],  # input feature dict
            s_inputs: torch.Tensor,  # (bs, n_tokens, c_token)
            s_trunk: torch.Tensor,  # (bs, n_tokens, c_token)
            z_trunk: torch.Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            sd_data: float = 16.0,
            mask: torch.Tensor = None,  # (bs, n_tokens)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffusion conditioning.
        Args:
            timesteps:
                [*, 1] timestep tensor
            features:
                input feature dictionary for the RelativePositionEncoding containing:
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
            s_inputs:
                [*, n_tokens, c_token] Single conditioning input
            s_trunk:
                [*, n_tokens, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, n_tokens, n_tokens, c_pair] Pair conditioning from Pairformer trunk
            sd_data:
                Scaling factor for the timesteps before fourier embedding
            mask:
                [*, n_tokens] token mask
        """
        # Pair conditioning
        pair_repr = torch.cat([z_trunk, self.relative_position_encoding(features, mask)], dim=-1)
        pair_repr = self.linear_pair(self.pair_layer_norm(pair_repr))
        for transition in self.pair_transitions:
            pair_repr = pair_repr + transition(pair_repr)

        # Single conditioning
        token_repr = torch.cat([s_trunk, s_inputs], dim=-1)
        token_repr = self.linear_single(self.single_layer_norm(token_repr))
        fourier_repr = self.fourier_embedding(torch.log(torch.div(torch.div(timesteps, sd_data), 4.0)))
        fourier_repr = self.linear_fourier(self.fourier_layer_norm(fourier_repr))
        token_repr = token_repr + fourier_repr.unsqueeze(1)
        for transition in self.single_transitions:
            token_repr = token_repr + transition(token_repr)

        # Mask outputs
        if mask is not None:
            token_repr = mask.unsqueeze(-1) * token_repr
            pair_mask = (mask[:, :, None] * mask[:, None, :]).unsqueeze(-1)  # (bs, n_tokens, n_tokens, 1)
            pair_repr = pair_repr * pair_mask

        return token_repr, pair_repr

    def forward(
            self,
            timesteps: torch.Tensor,  # timestep (bs, 1)
            features: Dict[str, torch.Tensor],  # input feature dict
            s_inputs: torch.Tensor,  # (bs, n_tokens, c_token)
            s_trunk: torch.Tensor,  # (bs, n_tokens, c_token)
            z_trunk: torch.Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            sd_data: float = 16.0,
            mask: torch.Tensor = None,  # (bs, n_tokens)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return checkpoint(self._forward, timesteps, features, s_inputs, s_trunk, z_trunk, sd_data, mask)
