import torch
from torch import nn, Tensor
from src.models.components.primitives import Linear, LinearNoBias
from src.models.pairformer import PairformerStack
from typing import Optional
from src.utils.tensor_utils import one_hot, add
from typing import Dict


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution as in AlphaFold2.

    For use in computation of distogram loss, subsection 1.9.8 of AlphaFold2 supplement.
    """

    def __init__(self, c_z: int, no_bins: int, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins
        self.linear = Linear(self.c_z, self.no_bins)

    def forward(self, z: Tensor) -> Tensor:  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class ConfidenceHead(nn.Module):
    def __init__(
            self,
            c_s: int,
            c_z: int,
            no_blocks: int = 4,
            no_bins_pde: int = 64,
            no_bins_plddt: int = 64,
            no_bins_pae: int = 64,
            c_hidden_mul: int = 128,
            c_hidden_pair_attn: int = 32,
            no_heads_tri_attn: int = 4,
            no_heads_single_attn: int = 16,
            transition_n: int = 4,
            pair_dropout: float = 0.25,
            fuse_projection_weights: bool = False,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
    ):
        super(ConfidenceHead, self).__init__()
        self.no_bins_pde = no_bins_pde
        self.no_bins_plddt = no_bins_plddt
        self.no_bins_pae = no_bins_pae

        # S_inputs projection
        self.linear_s_i = LinearNoBias(c_s, c_z)
        self.linear_s_j = LinearNoBias(c_s, c_z)

        self.linear_pair_dist = LinearNoBias(11, c_z)
        self.pairformer = PairformerStack(
            c_s=c_s,
            c_z=c_z,
            no_blocks=no_blocks,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_attn=c_hidden_pair_attn,
            no_heads_tri_attn=no_heads_tri_attn,
            no_heads_single_attn=no_heads_single_attn,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            blocks_per_ckpt=blocks_per_ckpt,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        self.pde_head = DistogramHead(c_z, no_bins=64)
        self.linear_plddt = LinearNoBias(c_s, no_bins_plddt)
        self.linear_p_resolved = LinearNoBias(c_s, 2)
        self.linear_pae = LinearNoBias(c_z, no_bins_pae)

    def forward(
            self,
            s_inputs: Tensor,  # (bs, n_tokens, c_s)
            s: Tensor,  # (bs, n_tokens, c_s)
            z: Tensor,  # (bs, n_tokens, n_tokens, c_z)
            x_repr: Tensor,  # (bs, n_tokens, 3)
            single_mask: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_flash: bool = False,
            inplace_safe: bool = False,
    ):
        """
        Args:
            s_inputs:
                [bs, n_tokens, c_s] input single representation from InputEmbedder
            s:
                [bs, n_tokens, c_s] single representation
            z:
                [bs, n_tokens, n_tokens, c_z] pair representation
            x_repr:
                [bs * samples_per_trunk, n_atoms, 3] predicted coordinates of representative atoms
            single_mask:
                [bs, n_tokens] single masking
            pair_mask:
                [bs, n_tokens, n_tokens] pair masking
            chunk_size:
                Inference-time sub-batch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_flash:
                Whether to use Flash attention within Pairformer.
                Can be used with use_deepspeed_evo_attention since DS4Science kernels are only used within
                triangular attention.
            inplace_safe:
                whether to use inplace ops
        Returns:
            output dictionary containing the logits (pre-softmax) for pLDDT, PAE, PDE,
            and experimentally resolved confidence measures.
        """
        # Grab data about the input
        dtype, device = s.dtype, s.device

        # Embed s_inputs
        z = add(z,
                self.linear_s_i(s_inputs[..., None, :, :]) + self.linear_s_j(s_inputs[..., :, None, :]),
                inplace=inplace_safe)

        # Embed pair distances of representative atoms
        d_ij = torch.sqrt(
            torch.sum(
                (x_repr[..., None, :, :] - x_repr[..., :, None, :]),
                dim=-1,
            ) ** 2)
        z = add(z,
                self.linear_pair_dist(
                    one_hot(d_ij, v_bins=torch.linspace(3.375, 21.375, steps=11, device=device, dtype=dtype))
                ),
                inplace=inplace_safe)

        # Run Pairformer
        s, z = self.pairformer(
            s=s,
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_flash=use_flash,
            inplace_safe=inplace_safe
        )

        # Project logits
        logits_pde = self.pde_head(z)
        logits_pae = self.linear_pae(z)

        logits_p_resolved = self.linear_p_resolved(s)
        # TODO: I learned how to compute this now, I should replace this with a proper implementation
        # N_max_atoms_per_token = 14
        logits_plddt = self.linear_plddt(s)

        output = {
            "plddt_logits": logits_plddt,  # (bs, n_tokens, no_bins_plddt)
            "pae_logits": logits_pae,  # (bs, n_tokens, n_tokens, no_bins_pae)
            "pde_logits": logits_pde,  # (bs, n_tokens, no_bins_pde)
            "experimentally_resolved_logits": logits_p_resolved  # (bs, n_tokens, 2)
        }

        return output

