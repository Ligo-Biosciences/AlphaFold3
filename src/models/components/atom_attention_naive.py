# Copyright 2024 Ligo Biosciences Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a naive implementation of the atom attention components. 
 We did early experiments with a PyTorch-native implementation that is supposed to use memory more efficiently, 
 but they did not show much benefit since attention implementations in PyTorch were much slower despite 
 adding considerable clutter and complexity. We fall back to the Deepspeed4Science optimized attention kernel, which reduce 
 the memory consumption to linear anyway. In practice, we only observe about a 20% increase in runtime. 
 The memory usage is approximately the same. 

This is not recommended for large scale training. 
The smart move here will be to migrate to FlexAttention once there is bias gradient support or to ScaleFold's kernels if they become available.
"""
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Dict, NamedTuple, Optional, Tuple
from src.models.components.primitives import Linear, LinearNoBias
from src.models.components.attention_pair_bias import AttentionPairBias
from torch.nn import LayerNorm
from src.models.components.transition import ConditionedTransitionBlock
from functools import lru_cache, partial
from src.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
checkpoint = get_checkpoint_fn()


class AtomTransformerBlock(nn.Module):
    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            no_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            inf: float = 1e8
    ):
        """Initialize a block within AtomTransformer module.
        Args:
            c_atom:
                Total dimension of the model.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
        """
        super().__init__()
        self.attention = AttentionPairBias(
            dim=c_atom,
            c_pair=c_atompair,
            no_heads=no_heads,
            dropout=dropout,
            input_gating=True,
            residual=False,
        )
        self.transition = ConditionedTransitionBlock(c_atom)
        self.inf = inf
        self.n_queries = n_queries
        self.n_keys = n_keys
    
    @lru_cache(maxsize=2)
    def _prep_betas(self, n_atoms: int, device: torch.device) -> torch.Tensor:
        """Prepare the betas that will be added to the attention scores for locality constraint."""
        # Create beta matrix filled with -inf
        beta = torch.full((n_atoms, n_atoms), -self.inf, device=device)

        # Calculate centers
        center_interval = 32
        centers = torch.arange(16.0, n_atoms, center_interval, device=device)

        # Vectorized operation to set valid attention regions to 0, invalid regions to -inf
        for c in centers:
            l_start = max(0, int(c - self.n_queries // 2))
            l_end = min(n_atoms, int(c + self.n_queries // 2))
            m_start = max(0, int(c - self.n_keys // 2))
            m_end = min(n_atoms, int(c + self.n_keys // 2))
            beta[l_start:l_end, m_start:m_end] = 0.0

        return beta

    def forward(
            self,
            atom_single: Tensor,
            atom_proj: Tensor,
            atom_pair: Tensor,
            mask: Optional[Tensor] = None,
            use_deepspeed_evo_attention: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Grab data about the input
        *_, n_atoms, _ = atom_single.shape

        # Compute betas
        betas = self._prep_betas(n_atoms, atom_single.device)  # (1, n_atoms, n_atoms)

        # AttentionPairBias
        atom_single = atom_single + self.attention(
            atom_single, atom_proj, atom_pair, mask, betas, 
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )
        # ConditionedTransitionBlock
        atom_single = atom_single + self.transition(atom_single, atom_proj)
        return atom_single, atom_proj, atom_pair
    

class AtomTransformer(nn.Module):
    """Implements the AtomTransformer"""

    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            no_blocks: int = 3,
            no_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
    ):
        """
        Initialize the AtomTransformer module.
        Args:
            c_atom:
                Total dimension of the model.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            blocks_per_ckpt:
                Number of AtomTransformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
        """
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.no_heads = no_heads
        self.no_blocks = no_blocks
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList(
            [AtomTransformerBlock(c_atom=c_atom,
                                  no_heads=no_heads,
                                  dropout=dropout,
                                  n_queries=n_queries,
                                  n_keys=n_keys,
                                  c_atompair=c_atompair)
             for _ in range(no_blocks)]
        )

    def _prep_blocks(
            self,
            atom_single: Tensor,
            atom_proj: Tensor,
            atom_pair: Tensor,
            mask: Optional[Tensor] = None,
            use_deepspeed_evo_attention: bool = True
    ):
        """Prepare the input tensors for each AtomTransformerBlock."""
        blocks = [
            partial(
                block,
                mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention
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
    

    def forward(
            self,
            atom_single: Tensor,
            atom_proj: Tensor,
            atom_pair: Tensor,
            mask: Optional[Tensor] = None,
            use_deepspeed_evo_attention: bool = True
    ):
        """
        Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement.
        Args:
            atom_single:
                [bs, S, n_atoms, c_atom] atom single representation where S is the samples per trunk dimension.
            atom_proj:
                [bs, n_atoms, c_atom] atom projection representation.
            atom_pair_local:
                [bs, n_atoms, n_atoms, c_atompair] atom pair representation tensor.
            mask:
                [bs, n_atoms] atom mask tensor where 1.0 indicates atom to be attended and
                0.0 indicates atom not to be attended. The mask is shared across the S dimension.
        """
        # Expand atom_proj for proper broadcasting
        atom_proj = atom_proj.unsqueeze(-3)

        blocks = self._prep_blocks(
            atom_single=atom_single,
            atom_proj=atom_proj,
            atom_pair=atom_pair,
            mask=mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        atom_single, atom_proj, atom_pair = checkpoint_blocks(
            blocks,
            args=(atom_single, atom_proj, atom_pair),
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return atom_single
    


def gather_token_repr(
        token_repr: Tensor,  # (bs, n_tokens, c_token)
        tok_idx: Tensor  # (bs, n_atoms)
) -> Tensor:
    """
    Gather token representations based on indices from tok_idx.

    Args:
        token_repr:
            [*, n_tokens, c_token] token representation
        tok_idx:
            [*, n_atoms] token indices.

    Returns:
        [batch_size, n_atoms, c_token] with gathered token embeddings.
    """
    batch_size, n_atoms = tok_idx.shape
    _, n_tokens, embed_dim = token_repr.shape
    tok_idx = tok_idx.long()  # convert to int for indexing

    # Expand tok_idx to have the same number of dimensions as token_repr
    new_shape = token_repr.shape[:-2] + (n_atoms, embed_dim)
    tok_idx_expanded = tok_idx.unsqueeze(-1).expand(new_shape)

    # Use torch.gather to gather embeddings from token_repr
    gathered_embeddings = torch.gather(
        token_repr,
        dim=-2,
        index=tok_idx_expanded
    )
    return gathered_embeddings


def aggregate_atom_to_token(
        atom_representation,  # (bs, S, n_atoms, c_atom)
        tok_idx: Tensor,  # (bs, n_atoms)
        n_tokens: int
) -> Tensor:
    """
    Aggregates atom representations to token representations.

    Args:
        atom_representation:
            The atom representations tensor of shape (bs, S, n_atoms, c_atom).
        tok_idx:
            The index tensor of shape (bs, n_atoms) indicating which token each atom belongs to.
        n_tokens (int):
            The number of tokens.
    Returns:
        Aggregated token representations of shape (bs, S, n_tokens, c_atom).
    Warning: this method is masking aware as long as tok_idx does not encode a mapping like
    masked_atom -> legitimate_token
    """
    bs, S, n_atoms, c_atom = atom_representation.shape
    tok_idx = tok_idx.long()  # convert to int for indexing

    # Initialize the token representation tensor with zeros
    token_representation = torch.zeros((bs, S, n_tokens, c_atom),
                                       device=atom_representation.device,
                                       dtype=atom_representation.dtype)

    # Expand tok_idx to make it compatible for scattering with atom_representation
    tok_idx_expanded = tok_idx[:, None, :, None].expand(-1, S, -1, c_atom)  # (bs, S, n_atoms, c_atom)

    # Aggregate atom representations into token representations
    token_representation = torch.scatter_reduce(
        token_representation,
        dim=-2,
        index=tok_idx_expanded,
        src=atom_representation,
        reduce='mean',
        include_self=False
    )
    return token_representation


def map_token_pairs_to_atom_pairs(
        token_pairs: torch.Tensor,  # (bs, n_tokens, c_pair)
        tok_idx: torch.Tensor  # (bs, n_atoms)
) -> torch.Tensor:
    """Given token pairs and token indices, map token pairs to atom pairs.
    Args:
        token_pairs (torch.Tensor):
            Tensor of shape (bs, n_tokens, n_tokens, c_pair).
        tok_idx (torch.Tensor):
            Tensor of shape (bs, n_atoms) containing token indices per atom.
    Returns:
        torch.Tensor: Tensor of shape (bs, n_atoms, n_atoms, c_pair) containing atom pair embeddings
        derived from token pair embeddings. For each atom pair (l, m), the corresponding token pair's
        embeddings are extracted.
    """
    bs, n_atoms = tok_idx.shape
    _, n_tokens, _, c_pair = token_pairs.shape

    # Expand tok_idx for efficient gather operation
    tok_idx_l = tok_idx.unsqueeze(2).expand(-1, -1, n_atoms)
    tok_idx_m = tok_idx.unsqueeze(1).expand(-1, n_atoms, -1)
    batch_index = torch.arange(bs, device=token_pairs.device).reshape(bs, 1, 1)

    # Gather token pair embeddings using advanced indexing
    atom_pairs = token_pairs[batch_index, tok_idx_l, tok_idx_m, :]

    return atom_pairs


class AtomAttentionEncoderOutput(NamedTuple):
    """Structured output class for AtomAttentionEncoder."""
    token_single: torch.Tensor  # (bs, n_tokens, c_token)
    atom_single_skip_repr: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_single_skip_proj: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_pair_skip_repr: torch.Tensor  # (bs, n_atoms, n_atoms c_atompair)


class AtomAttentionEncoder(nn.Module):
    def __init__(
            self,
            c_token: int,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 16,
            no_blocks: int = 3,
            no_heads: int = 4,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            trunk_conditioning: bool = False,
            clear_cache_between_blocks: bool = False
    ):
        """Initialize the AtomAttentionEncoder module.
            Args:
                c_token:
                    The number of channels for the token representation.
                c_atom:
                    The number of channels for the atom representation. Defaults to 128.
                c_atompair:
                    The number of channels for the pair representation. Defaults to 16.
                c_trunk_pair:
                    The number of channels for the trunk pair representation. Defaults to 16.
                no_blocks:
                    Number of blocks in AtomTransformer. Defaults to 3.
                no_heads:
                    Number of parallel attention heads. Note that c_atom will be split across no_heads
                    (i.e. each head will have dimension c_atom // no_heads).
                dropout:
                    Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
                n_queries:
                    The size of the atom window. Defaults to 32.
                n_keys:
                    Number of atoms each atom attends to in local sequence space. Defaults to 128.
                trunk_conditioning:
                    Whether to condition the atom single and atom-pair representation on the trunk.
                    Defaults to False.
                clear_cache_between_blocks:
                    Whether to clear CUDA's GPU memory cache between blocks of the
                    stack. Slows down each block but can reduce fragmentation
        """
        super().__init__()
        self.no_blocks = no_blocks
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_trunk_pair = c_trunk_pair
        self.no_heads = no_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.trunk_conditioning = trunk_conditioning
        self.clear_cache_between_blocks = clear_cache_between_blocks

        # Embedding per-atom metadata, concat(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)
        self.linear_atom_embedding = LinearNoBias(3 + 1 + 1 + 4 + 4, c_atom)  # 128, * 64

        # Embedding offsets between atom reference positions
        self.linear_atom_offsets = LinearNoBias(3, c_atompair)
        self.linear_atom_distances = LinearNoBias(1, c_atompair)

        # Embedding the valid mask
        self.linear_mask = LinearNoBias(1, c_atompair)

        if trunk_conditioning:
            self.proj_trunk_single = nn.Sequential(
                LayerNorm(c_token),
                LinearNoBias(c_token, c_atom)
            )
            self.proj_trunk_pair = nn.Sequential(
                LayerNorm(c_trunk_pair),
                LinearNoBias(c_trunk_pair, c_atompair)
            )

            self.linear_noisy_pos = LinearNoBias(3, c_atom)

        # Adding the single conditioning to the pair representation
        self.linear_single_to_pair_row = LinearNoBias(c_atom, c_atompair, init='relu')
        self.linear_single_to_pair_col = LinearNoBias(c_atom, c_atompair, init='relu')

        # Small MLP on the pair activations
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair, init='relu'),
            nn.ReLU(),
            LinearNoBias(c_atompair, c_atompair, init='final')
        )

        # Cross attention transformer
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_blocks=no_blocks,
            no_heads=no_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            clear_cache_between_blocks=clear_cache_between_blocks
        )

        # Final linear
        self.output_proj = nn.Sequential(
            LinearNoBias(c_atom, c_token, init='relu'),
            nn.ReLU()
        )

    def init_pair_repr(
            self,
            features: Dict[str, Tensor],
            atom_cond: Tensor,
            z_trunk: Optional[Tensor],
    ) -> Tensor:
        """Compute the pair representation for the atom transformer.
        This is done in a separate function for checkpointing. The intermediate activations due to the
        atom pair representations are large and can be checkpointed to reduce memory usage.
        Args:
            features:
                Dictionary of input features.
            atom_cond:
                [bs, n_atoms, c_atom] The single atom conditioning from init_single_repr
            z_trunk:
                [bs, n_tokens, n_tokens, c_trunk] the pair representation from the trunk
        Returns:
            [bs, n_atoms, n_atoms, c_atompair] The pair representation
        """
        # Embed offsets between atom reference positions
        offsets = features['ref_pos'][..., None, :] - features['ref_pos'][..., None, :, :]  # (bs, n_atoms, n_atoms, 3)
        valid_mask = features['ref_space_uid'][..., :, None] == features['ref_space_uid'][..., None, :]
        valid_mask = valid_mask.unsqueeze(-1).to(offsets.dtype)  # convert boolean to binary where 1.0 is True, 0.0 is False
        atom_pair = self.linear_atom_offsets(offsets) * valid_mask

        # Embed pairwise inverse squared distances, and the valid mask
        squared_distances = offsets.pow(2).sum(dim=-1, keepdim=True)  # (bs, n_atoms, n_atoms, 1)
        inverse_dists = torch.reciprocal(torch.add(squared_distances, 1))
        atom_pair = atom_pair + self.linear_atom_distances(inverse_dists) * valid_mask
        atom_pair = atom_pair + self.linear_mask(valid_mask) * valid_mask

        # If provided, add trunk embeddings
        if self.trunk_conditioning:
            atom_pair = atom_pair + map_token_pairs_to_atom_pairs(
                self.proj_trunk_pair(z_trunk),
                features['atom_to_token']
            )

        # Add the combined single conditioning to the pair representation
        atom_pair = self.linear_single_to_pair_row(F.relu(atom_cond[:, None, :, :])) + \
                    self.linear_single_to_pair_col(F.relu(atom_cond[:, :, None, :])) + atom_pair
        
        # Run a small MLP on the pair activations
        atom_pair = atom_pair + self.pair_mlp(atom_pair)
        return atom_pair
    
    def init_single_repr(
            self,
            features: Dict[str, Tensor],
            s_trunk: Optional[Tensor],
            noisy_pos: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute the single representation for the atom transformer.
        Args:
            features:
                Dictionary of input features.
            s_trunk:
                [*, n_tokens, c_token] the token representation from the trunk
            noisy_pos:
                [*, S, n_atoms, 3] the noisy atom positions where S is the
                samples_per_trunk dimension.
        Returns:
            - atom_single:
                [*, S, n_atoms, c_atom] atom single representation
            - atom_single_conditioning:
                [*, n_atoms, c_atom] atom single conditioning representation
        """
        batch_size, n_atoms, _ = features['ref_pos'].size()

        # Embed atom metadata
        atom_single_conditioning = self.linear_atom_embedding(
            torch.cat(
                [features['ref_pos'],
                 features['ref_charge'].unsqueeze(-1),
                 features['ref_mask'].unsqueeze(-1),
                 features['ref_element'],
                 features['ref_atom_name_chars'].reshape(batch_size, n_atoms, 4)],  # * 64
                dim=-1
            )
        )
        # Initialize the atom single representation as the single conditioning
        atom_single = atom_single_conditioning.clone()
        # atom_single_conditioning -> c_l in AF3 Supplement
        # atom_single -> q_l in AF3 Supplement

        # Add the samples_per_trunk dimension
        atom_single = atom_single.unsqueeze(-3)  # [*, 1, n_atoms, c_atom]

        # If provided, add trunk embeddings and noisy positions
        if self.trunk_conditioning:
            atom_single_conditioning = atom_single_conditioning + gather_token_repr(
                self.proj_trunk_single(s_trunk),
                features['atom_to_token']
            )
            # Add the noisy positions
            atom_single = atom_single + self.linear_noisy_pos(noisy_pos)  # [*, S, n_atoms, c_atom]

        return atom_single, atom_single_conditioning
    
    def forward(
            self,
            features: Dict[str, Tensor],
            n_tokens: int,
            s_trunk: Optional[Tensor] = None,  # (bs, n_tokens, c_token)
            z_trunk: Optional[Tensor] = None,  # (bs, n_tokens, c_trunk_pair)
            noisy_pos: Optional[Tensor] = None,  # (bs, S, n_atoms, 3)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
            use_deepspeed_evo_attention: bool = True
    ) -> AtomAttentionEncoderOutput:
        """Forward pass for the AtomAttentionEncoder module.
        Args:
            features:
                Dictionary containing the input features:
                    "ref_pos":
                        [*, N_atoms, 3] atom positions in the reference conformers, with
                        a random rotation and translation applied. Atom positions in Angstroms.
                    "ref_charge":
                        [*, N_atoms] Charge for each atom in the reference conformer.
                    "ref_mask":
                        [*, N_atoms] Mask indicating which atom slots are used in the reference
                        conformer.
                    "ref_element":
                        [*, N_atoms, 128] One-hot encoding of the element atomic number for each atom
                        in the reference conformer, up to atomic number 128.
                    "ref_atom_name_chars":
                        [*, N_atom, 4, 64] One-hot encoding of the unique atom names in the reference
                        conformer. Each character is encoded as ord(c - 32), and names are padded to
                        length 4.
                    "ref_space_uid":
                        [*, N_atoms] Numerical encoding of the chain id and residue index associated
                        with this reference conformer. Each (chain id, residue index) tuple is assigned
                        an integer on first appearance.
                    "atom_to_token":
                        [*, N_atoms] Token index for each atom in the flat atom representation.
            n_tokens:
                The number of tokens that will be in the output representation.
            s_trunk:
                [*, N_tokens, c_token] single representation of the Pairformer trunk
            z_trunk:
                [*, N_tokens, N_tokens, c_pair] pair representation of the Pairformer trunk
            noisy_pos:
                [*, S, N_atoms, 3] Tensor containing the noisy positions. Defaults to None.
            mask:
                [*, N_atoms]
        Returns:
            A namedtuple containing the following fields:
                token_single:
                    [*, N_tokens, c_token] single representation
                atom_single_skip_repr:
                    [*, S, N_atoms, c_atom] atom single representation (denoted q_l in AF3 Supplement)
                atom_single_skip_proj:
                    [*, N_atoms, c_atom] atom single projection (denoted c_l in AF3 Supplement)
                atom_pair_skip_repr:
                    [*, N_atoms, n_atoms, c_atompair] atom pair representation
                    (denoted p_lm in AF3 Supplement)
        """
        # Initialize representations
        atom_single, atom_single_conditioning = self.init_single_repr(features, s_trunk, noisy_pos)
        atom_pair = checkpoint(self.init_pair_repr, features, atom_single_conditioning, z_trunk)

        # Cross attention transformer
        atom_single = self.atom_transformer(atom_single, atom_single_conditioning, atom_pair, mask, use_deepspeed_evo_attention)

        # Aggregate per-atom representation to per-token representation
        token_repr = aggregate_atom_to_token(
            atom_representation=self.output_proj(atom_single),
            tok_idx=features['atom_to_token'],
            n_tokens=n_tokens
        )  # (*, S, N_tokens, c_atom)

        output = AtomAttentionEncoderOutput(
            token_single=token_repr,  # (*, S, N_tokens, c_atom)
            atom_single_skip_repr=atom_single,  # (*, N_atoms, c_atom)
            atom_single_skip_proj=atom_single_conditioning,  # (*, N_atoms, c_atom)
            atom_pair_skip_repr=atom_pair,  # (bs, n_atoms, n_atoms, c_atompair)
        )
        return output


class AtomAttentionDecoder(nn.Module):
    def __init__(
            self,
            c_token: int,
            c_atom: int = 128,
            c_atompair: int = 16,
            no_blocks: int = 3,
            no_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
    ):
        """Initialize the AtomAttentionDecoder module.
        Args:
            c_token:
                The number of channels for the token representation.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.
            no_blocks:
                Number of blocks.
            no_heads:
                Number of parallel attention heads. Note that c_atom will be split across num_heads
                (i.e. each head will have dimension c_atom // num_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
        """
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.no_blocks = no_blocks
        self.no_heads = no_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_blocks=no_blocks,
            no_heads=no_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
        )

        self.linear_atom = Linear(c_token, c_atom, init='default', bias=False)
        self.linear_update = Linear(c_atom, 3, init='final', bias=False)
        self.layer_norm = nn.LayerNorm(c_atom)

    def forward(
            self,
            token_repr,  # (bs, n_tokens, c_token)
            atom_single_skip_repr,  # (bs, n_atoms, c_atom)
            atom_single_skip_proj,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx,  # (bs, n_atoms)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
            use_deepspeed_evo_attention: bool = True
    ):
        """
        AtomAttentionDecoder. Algorithm 6 in AlphaFold3 supplement.
            Args:
                token_repr:
                    Per-token activations. Shape (bs, n_tokens, c_atom).
                atom_single_skip_repr:
                    Per-atom activations added as the skip connection. Shape (bs, n_atoms, c_atom).
                atom_single_skip_proj:
                    Per-atom activations provided to AtomTransformer.
                atom_pair_skip_repr:
                    Pair activations provided to AtomTransformer. Shape (bs, n_atoms, n_atoms, c_atom).
                tok_idx:
                    Token indices that encode which token each atom belongs to.  Shape (bs, n_atoms).
                mask:
                    Mask for the atom transformer. Shape (bs, n_atoms).
        Returns:
            a tensor of per-atom coordinate updates. Shape (bs, n_atoms, 3).
        """
        # Broadcast per-token activations to per-atom activations and add the skip connection
        bs, S, n_tokens, c_atom = token_repr.shape
        atom_single = self.linear_atom(  # vectorize to pretend S is the batch dimension for the gather op
            torch.vmap(gather_token_repr)(token_repr, tok_idx.unsqueeze(-2).expand(-1, S, -1))
        )
        atom_single = atom_single + atom_single_skip_repr  # (bs, S, n_atoms, c_atom)

        # Cross-attention transformer
        atom_single_repr = self.atom_transformer(atom_single, atom_single_skip_proj, atom_pair_skip_repr, mask, use_deepspeed_evo_attention)

        # Map to positions update
        r_atom_update = self.linear_update(self.layer_norm(atom_single_repr))
        return r_atom_update

