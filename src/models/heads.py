import torch
from torch import nn, Tensor
from src.models.components.primitives import Linear, LinearNoBias
from src.models.pairformer import PairformerStack
from typing import Optional
from src.utils.tensor_utils import one_hot, add


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

        self.linear = Linear(self.c_z, self.no_bins, init="final")

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
            no_bins_pae: int = 64
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
            single_mask: Optional[Tensor] = None,  # (bs, n_tokens)
            pair_mask: Optional[Tensor] = None,  # (bs, n_tokens, n_tokens)
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ):
        """
        Args:
            s_inputs:
                [*, n_tokens, c_s] input single representation from InputEmbedder
            s:
                [*, n_tokens, c_s] single representation
            z:
                [*, n_tokens, n_tokens, c_z] pair representation
            x_repr:
                [*, n_tokens, 3] predicted coordinates of representative atoms
            single_mask:
                [*, n_tokens] mask for the single representation
            pair_mask:
                [*, n_tokens, n_tokens] mask for the pair representation
            chunk_size:
                Inference-time sub-batch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
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
            use_lma=use_lma,
            inplace_safe=inplace_safe
        )

        # Project logits
        logits_pde = self.pde_head(z)
        logits_pae = self.linear_pae(z)

        # TODO: the pseudocode is ambiguous about how these are computed.
        #  I will simply project them from the single representation.
        logits_plddt = self.linear_plddt(s)
        logits_p_resolved = self.linear_p_resolved(s)

        output = {
            "logits_plddt": logits_plddt,  # (bs, n_tokens, no_bins_plddt)
            "logits_pae": logits_pae,  # (bs, n_tokens, n_tokens, no_bins_pae)
            "logits_pde": logits_pde,  # (bs, n_tokens, no_bins_pde)
            "logits_p_resolved": logits_p_resolved  # (bs, n_tokens, 2)
        }

        return output

