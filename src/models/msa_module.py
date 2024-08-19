"""The MSA Module in AlphaFold3 fulfills a similar role to the Extra MSA Stack in AlphaFold2 and hence a fairly
similar architecture to AlphaFold-Multimer in the block. It samples a new i.i.d. random subset of the MSA for each
recycling iteration, the MSA sequences and x features then get embedded into representation m_si for each token n
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
from torch.nn import LayerNorm
from src.models.components.primitives import LinearNoBias
from src.models.pairformer import PairStack
from src.models.components.outer_product_mean import OuterProductMean
from src.models.components.transition import Transition
from src.models.components.dropout import DropoutRowwise
from src.utils.tensor_utils import add
from functools import partial
from typing import Dict
from src.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
checkpoint = get_checkpoint_fn()


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
        split_heads = nn.Unflatten(dim=-1, unflattened_size=(no_heads, c_hidden))
        self.msa_proj = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='glorot'),
            split_heads  # split the heads
        )
        self.to_gamma = nn.Sequential(
            LinearNoBias(c_msa, c_hidden * no_heads, init='gating'),
            split_heads,  # split the heads
            nn.Sigmoid()
        )

        # Pair
        self.proj_pair_bias = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, no_heads, init="normal")
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
        Returns:
            [*, N_seq, N_res, C_m] updated MSA representation
        TODo: use the optimized triton kernel from FastFold here!
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
        new_v_shape = (v.shape[:-4] + (n_seq, n_res, n_res, self.no_heads, self.c_hidden))
        v = v.unsqueeze(-4).expand(new_v_shape)  # (*, seq, res, res, heads, c_hidden)

        # Weighted average with gating
        weights = self.softmax(b)
        weights = weights.unsqueeze(-4).unsqueeze(-1)  # (*, 1, res, res, heads, 1)
        o = g * torch.sum(v * weights, dim=-3)  # (*, seq, res, heads, c_hidden)

        # Output projection
        output = self.output_proj(
            o.reshape((m.shape[:-1] + (self.c_hidden * self.no_heads,)))  # (*, seq, res, c_hidden * heads)
        )
        return output


class MSAStack(nn.Module):
    """MSA stack that applies pair weighted averaging, dropout, and transition."""

    def __init__(
            self,
            c_msa: int,
            c_z: int,
            c_hidden: int = 8,
            no_heads: int = 8,
            dropout: float = 0.15,
            inf: float = 1e8
    ):
        super(MSAStack, self).__init__()

        self.msa_pair_avg = MSAPairWeightedAveraging(
            c_msa=c_msa,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf
        )
        self.dropout_row_layer = DropoutRowwise(dropout)
        self.transition = Transition(c_msa)

    def forward(
            self,
            m: Tensor,
            z: Tensor,
            msa_mask: Optional[Tensor] = None,
            z_mask: Optional[Tensor] = None,
            inplace_safe: bool = False
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
            inplace_safe:
                whether to perform ops inplace
        Returns:
            [*, N_seq, N_res, C_m] updated MSA representation
        """
        m = add(
            m,
            self.dropout_row_layer(
                self.msa_pair_avg(
                    m=m,
                    z=z,
                    msa_mask=msa_mask,
                    z_mask=z_mask
                )
            ),
            inplace=inplace_safe
        )
        m = add(m, self.transition(m), inplace=inplace_safe)
        return m


class MSAModuleBlock(nn.Module):
    def __init__(
            self,
            c_msa: int = 64,
            c_z: int = 128,
            c_hidden: int = 32,
            no_heads: int = 8,
            c_hidden_tri_mul: int = 128,
            c_hidden_pair_attn: int = 32,
            no_heads_tri_attn: int = 4,
            transition_n: int = 4,
            pair_dropout: float = 0.25,
            fuse_projection_weights: bool = False,
            inf: float = 1e8
    ):
        super(MSAModuleBlock, self).__init__()

        # Communication
        self.outer_product_mean = OuterProductMean(c_m=c_msa, c_z=c_z, c_hidden=c_hidden)

        # MSA stack
        self.msa_stack = MSAStack(
            c_msa=c_msa,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=no_heads,
            inf=inf
        )

        # Pair stack
        self.pair_stack = PairStack(
            c_z=c_z,
            c_hidden_tri_mul=c_hidden_tri_mul,
            c_hidden_pair_attn=c_hidden_pair_attn,
            no_heads_tri_attn=no_heads_tri_attn,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf
        )

    def forward(
            self,
            m: Tensor,
            z: Tensor,
            msa_mask: Tensor,
            z_mask: Tensor,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            inplace_safe: bool = False,
    ):
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
            chunk_size:
                chunk size
            use_deepspeed_evo_attention:
                whether to use Deepspeed's optimized kernels for attention
            inplace_safe:
                whether to perform ops inplace
        Returns:
            Tuple of:
                [*, N_seq, N_res, C_m] updated MSA representation,
                [*, N_res, N_res, C_z] updated pair representation
        """
        # Communication
        z = add(
            z,
            self.outer_product_mean(
                m,
                mask=msa_mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
            ),
            inplace=inplace_safe
        )

        # MSA stack
        m = self.msa_stack(
            m=m,
            z=z,
            msa_mask=msa_mask,
            z_mask=z_mask,
            inplace_safe=inplace_safe
        )

        # Pair stack
        z = self.pair_stack(
            z=z,
            pair_mask=z_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            inplace_safe=inplace_safe,
        )
        return m, z


class MSAModule(nn.Module):
    def __init__(
            self,
            no_blocks: int = 4,
            c_msa: int = 64,
            c_token: int = 384,
            c_z: int = 128,
            c_hidden: int = 32,
            no_heads: int = 8,
            c_hidden_tri_mul: int = 128,
            c_hidden_pair_attn: int = 32,
            no_heads_tri_attn: int = 4,
            transition_n: int = 4,
            pair_dropout: float = 0.25,
            fuse_projection_weights: bool = False,
            clear_cache_between_blocks: bool = False,
            blocks_per_ckpt: int = 1,
            inf: float = 1e8
    ):
        """
        Initialize the MSA module.
        Args:
            no_blocks:
                number of MSAModuleBlocks
            c_msa:
                MSA representation dim
            c_token:
                Single representation dim
            c_z:
                pair representation dim
            c_hidden:
                hidden representation dim
            no_heads:
                number of heads in the pair averaging
            c_hidden_tri_mul:
                hidden dimensionality of triangular multiplicative updates
            c_hidden_pair_attn:
                hidden dimensionality of triangular attention
            no_heads_tri_attn:
                number of heads in triangular attention
            transition_n:
                multiplication factor for the hidden dim during the transition
            pair_dropout:
                dropout rate within the pair stack
            fuse_projection_weights:
                whether to use FusedTriangleMultiplicativeUpdate or not
            blocks_per_ckpt:
                Number of blocks per checkpoint. If None, no checkpointing is used.
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super(MSAModule, self).__init__()
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_msa=c_msa,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                c_hidden_tri_mul=c_hidden_tri_mul,
                c_hidden_pair_attn=c_hidden_pair_attn,
                no_heads_tri_attn=no_heads_tri_attn,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf)
            for _ in range(no_blocks)
        ])

        # MSA featurization
        self.linear_msa_feat = LinearNoBias(49, c_msa)
        self.proj_s_inputs = LinearNoBias(c_token, c_msa, init='final')
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

    def _prep_blocks(
            self,
            msa_mask: Optional[Tensor] = None,
            z_mask: Optional[Tensor] = None,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            inplace_safe: bool = False,
    ):
        blocks = [
            partial(
                block,
                msa_mask=msa_mask,
                z_mask=z_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                inplace_safe=inplace_safe
            )
            for block in self.blocks
        ]

        # Clear CUDA's GPU memory cache between blocks
        if self.clear_cache_between_blocks:
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        return blocks

    def init_msa_repr(
            self,
            feats: Dict[str, Tensor],
            s_inputs: Tensor,
            msa_mask: Optional[Tensor] = None,
            inplace_safe: bool = False,
    ) -> Tensor:
        """Initializes the MSA representation."""
        msa_feats = feats["msa_feat"]
        # torch.cat([
        #    feats["msa"],
        #    feats["has_deletion"][..., None],
        #    feats["deletion_value"][..., None]],
        #    dim=-1)
        m = self.linear_msa_feat(msa_feats)
        m = add(m,
                self.proj_s_inputs(s_inputs[..., None, :, :]),
                inplace=inplace_safe)
        if msa_mask is not None:
            m = m * msa_mask[..., None]
        return m

    def forward(
            self,
            feats: Dict[str, Tensor],
            z: Tensor,
            s_inputs: Tensor,
            z_mask: Optional[Tensor] = None,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            inplace_safe: bool = False,
    ) -> Tensor:
        """
        Args:
            feats:
                Dictionary containing the MSA features with the following features:
                    "msa_feat":
                        [*, N_msa, N_token, 49] Concatenated MSA features from AF2
                    "msa_mask":
                        [*, N_seq, N_token] MSA mask
            z:
                [*, N_token, N_token, C_z] pair embeddings
            s_inputs:
                [*, N_token, c_token] single input embeddings
            z_mask:
                [*, N_token, N_token] pair mask
            chunk_size:
                chunk size
            use_deepspeed_evo_attention:
                whether to use Deepspeed's optimized kernels for attention
            inplace_safe:
                whether to perform ops inplace
        """
        # Prep MSA mask
        msa_mask = feats["msa_mask"]

        # Prep the blocks
        blocks = self._prep_blocks(
            msa_mask=msa_mask,
            z_mask=z_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            inplace_safe=inplace_safe
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        # Initialize the MSA embedding
        m = checkpoint(self.init_msa_repr, feats, s_inputs, msa_mask, inplace_safe)

        # Run with grad checkpointing
        m, z = checkpoint_blocks(
            blocks,
            args=(m, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return z


