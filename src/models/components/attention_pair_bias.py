import torch
from torch import nn
from torch.nn import functional as F
from src.models.components.primitives import (
    Linear, LinearNoBias, AdaLN, LayerNorm, Attention
)
from typing import Optional
from einops import rearrange


class AttentionPairBias(nn.Module):
    """Full self-attention with pair bias."""

    def __init__(
            self,
            dim: int,
            c_pair: int = 16,
            no_heads: int = 8,
            dropout: float = 0.0,
            input_gating: bool = True,
            residual: bool = True,
            inf: float = 1e8,
    ):
        """Initialize the AttentionPairBias module.
        Args:
            dim:
                Total dimension of the model.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            residual:
                Whether the module is used as a residual block. Default: True. This affects the initialization
                of the final projection layer of the MHA attention.
            input_gating:
                Whether the single representation should be gated with another single-like representation using
                adaptive layer normalization. Default: True.
        """
        super().__init__()
        self.dim = dim
        self.c_pair = c_pair
        self.num_heads = no_heads
        self.dropout = dropout
        self.input_gating = input_gating
        self.inf = inf

        # Perform check for dimensionality
        assert dim % no_heads == 0, f"the model dimensionality ({dim}) should be divisible by the " \
                                    f"number of heads ({no_heads}) "
        # Projections
        self.input_proj = None
        self.output_proj_linear = None
        if input_gating:
            self.input_proj = AdaLN(dim)

            # Output projection from AdaLN
            self.output_proj_linear = Linear(dim, dim, init='gating')
            self.output_proj_linear.bias = nn.Parameter(torch.ones(dim) * -2.0)  # gate values will be ~0.11
        else:
            self.input_proj = LayerNorm(dim)

        # Attention
        self.attention = Attention(
            c_q=dim,  # TODO: this Q needs to be projected from a linear layer with bias
            c_k=dim,
            c_v=dim,
            c_hidden=dim // no_heads,
            no_heads=no_heads,
            gating=True,
            residual=residual,
        )

        # Pair bias
        self.proj_pair_bias = nn.Sequential(
            LayerNorm(self.c_pair),
            LinearNoBias(self.c_pair, self.num_heads, init='normal')
        )

    def _prep_biases(
            self,
            single_repr: torch.Tensor,  # (*, S, N, c_s)
            pair_repr: torch.Tensor,  # (*, N, N, c_z)
            mask: Optional[torch.Tensor] = None,  # (*, N)
    ):
        """Prepares the mask and pair biases in the shapes expected by the DS4Science attention.

        Expected shapes for the DS4Science kernel:
        # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
        # res_mask: [Batch, N_seq, 1, 1, N_res]
        # pair_bias: [Batch, 1, Head, N_res, N_res]
        """
        # Compute the single mask
        n_seq, n_res, _ = single_repr.shape[-3:]
        if mask is None:
            # [*, N_seq, N_res]
            mask = single_repr.new_ones(
                single_repr.shape[:-3] + (n_seq, n_res),
            )
        else:
            # Expand mask by N_seq (or samples per trunk)
            new_shape = (mask.shape[:-1] + (n_seq, n_res))  # (*, N_seq, N_res)
            mask = mask.unsqueeze(-2).expand(new_shape)
            mask = mask.to(single_repr.dtype)
        # [*, N_seq, 1, 1, N_res]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # Project pair biases per head from pair representation
        pair_bias = self.proj_pair_bias(pair_repr)  # (bs, n_tokens, n_tokens, n_heads)
        pair_bias = rearrange(pair_bias, 'b n1 n2 h -> b h n1 n2')  # # (bs, h, n, n)
        pair_bias = pair_bias.unsqueeze(-4)
        return mask_bias, pair_bias

    def forward(
            self,
            single_repr: torch.Tensor,  # (*, S, N, c_s)
            single_proj: Optional[torch.Tensor] = None,  # (*, N, c_s)
            pair_repr: torch.Tensor = None,  # (*, N, N, c_z)
            mask: Optional[torch.Tensor] = None,  # (*, N)
            use_deepspeed_evo_attention: bool = False,
    ) -> torch.Tensor:
        """Full self-attention at the token-level with pair bias.
        The DS4Science kernel for MSA row-wise attention is re-purposed here for an efficient
        implementation of attention pair bias. The AttentionPairBias class is used in two
        main model components: the Pairformer and the Diffusion module. The main advantage of the
        kernel is in being able to accommodate a secondary batch-like dimension. In AlphaFold2, this
        is for N_seq in the MSA representation. In AlphaFold3, this is not needed in the Pairformer
        because we are using a single representation, so N_seq always equals 1. However, this is
        very useful in the diffusion module as multiple versions of the same input are created, and the
        same bias has to be added to this expanded representation throughout the DiffusionTransformer blocks.
        Here, we can use the N_seq dimension to host the samples per trunk which would make for a very memory
        efficient representation.
        Args:
            single_repr:
                [*, S, N, c_s] single representation, where S is the samples_per_trunk dimension.
            single_proj:
                [*, S, N, c_s] or [*, 1, N, c_s] single projection
            pair_repr:
                [*, N, c_z] pair representation
            mask:
                [*, N] attention mask where 1.0 indicates valid token, 0.0 indicates invalid token.
            use_deepspeed_evo_attention:
                Whether to use deepspeed attention or not.
        """

        # Input projection
        if self.input_gating:
            a = self.input_proj(single_repr, single_proj)  # AdaLN(a, s)  shape: (bs, S, n_tokens, c_atom)
        else:
            a = self.input_proj(single_repr)

        # Compute the biases
        mask_bias, pair_bias = self._prep_biases(single_repr, pair_repr, mask)

        # Attention
        output = self.attention(
            q_x=a,
            kv_x=a,
            biases=[mask_bias, pair_bias],
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )  # (bs, S, n_tokens, c_atom)

        # Output projection (from adaLN-Zero)
        if self.input_gating:
            output = F.sigmoid(self.output_proj_linear(output)) * output
        return output
