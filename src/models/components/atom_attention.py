"""Sequence-local atom attention.
The 'sequence-local atom attention' represents the whole structure as a flat list of atoms and allows all atoms to
'talk' directly to each other within a certain sequence neighbourhood. e.g. each subset of 32 atoms attends to the
subset of the nearby 128 atoms (nearby in the sequence space). This gives the network the capacity to learn general
rules about local atom constellations, independently of the coarse-grained tokenization where each standard residue
is represented with a single token only."""

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from src.models.components.primitives import AdaLN, Linear, LinearNoBias, Attention, compute_pair_attention_mask
from src.models.components.transition import ConditionedTransitionBlock
from typing import Dict, NamedTuple, Optional, Tuple
from functools import partial
from src.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
checkpoint = get_checkpoint_fn()


def partition_tensor(
        x: Tensor,  # (batch_size, n_atoms, c)
        n_queries: int = 32,
        n_keys: int = 128
) -> Tensor:
    """Partitions the input flat atom tensor into windows of n_keys with a slide stride of n_queries.
    The input tensor is padded to make the centers of the partitions align with the subset centers in AlphaFold3.
    Subset centers = (15.5, 47.5, 79.5, ...)
    """
    # Pad
    pad = n_keys // 2 - n_queries // 2
    x = pad_column(x, (pad, pad))

    # Sliding window along n_atoms dimension
    windows = x.unfold(dimension=1, size=n_keys, step=n_queries)
    return windows.transpose(-1, -2)  # unfold reverses channel dimension, undo this


def pad_column(x, pad, mode='constant', value=None) -> Tensor:
    """Applies padding to the second to last dimension of a tensor."""
    return F.pad(x.transpose(-1, -2), pad, mode=mode, value=value).transpose(-1, -2)


def extract_local_biases(bias_tensor: Tensor, n_queries: int = 32, n_keys: int = 128) -> Tensor:
    """Extracts biases that are local in the sequence space. Also pads the local biases with large negative values
    to mask the gaps during attention computation.
        Args:
            bias_tensor:
                A tensor of shape [batch_size, N_atoms, N_atoms, channels].
            n_queries:
                The increment between the centers of the partitions.
            n_keys:
                The length of the partitions.
        Returns:
            A tensor of shape [batch_size, N_atoms // partition_increment, partition_length, channels].
    """
    batch_size, n_atoms, _, channels = bias_tensor.shape
    # Pad bias tensor column-wise by n_keys // 2 - n_queries // 2 on each side
    pad = n_keys // 2 - n_queries // 2
    bias_tensor = pad_column(bias_tensor, (pad, pad), mode='constant', value=-1e4)

    # Compute the number of blocks along the first dimension
    num_blocks = n_atoms // n_queries

    # Initialize a list to store the result
    local_biases = []

    # Extract blocks and populate the result tensor
    for i in range(num_blocks):
        start_row = i * n_queries
        end_row = start_row + n_queries
        # We can stride along columns by n_keys to achieve overlaps
        start_col = i * n_queries
        end_col = start_col + n_keys

        local_biases.append(bias_tensor[:, start_row:end_row, start_col:end_col, :])
    return torch.stack(local_biases, dim=1)


class AtomAttentionPairBias(nn.Module):
    """Implements the sequence-local atom attention with pair bias.
    This is implemented separately to the attention module that performs full self-attention
    since sequence-local atom attention requires a memory-efficient implementation.
    """

    def __init__(
            self,
            c_atom: int = 128,
            num_heads=8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            c_atompair: int = 16,
    ):
        """Initialize the AtomAttentionPairBias module.
        Args:
            c_atom:
                Total dimension of the model.
            num_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.

        """
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys

        # Projections
        self.ada_ln = AdaLN(c_atom)
        self.output_proj_linear = Linear(c_atom, c_atom, init='gating')
        self.output_proj_linear.bias = nn.Parameter(torch.ones(c_atom) * -2.0)  # gate values will be ~0.11

        # Attention
        self.attention = Attention(
            c_q=c_atom,
            c_k=c_atom,
            c_v=c_atom,
            c_hidden=c_atom // num_heads,
            no_heads=num_heads,
            gating=True
        )

        # Pair bias
        self.layer_norm_pair = nn.LayerNorm(self.c_atompair)
        self.linear_pair = LinearNoBias(self.c_atompair, self.num_heads, init='default')

    def forward(
            self,
            atom_single_repr: Tensor,  # (bs, n_atoms, c_atom)
            atom_single_proj: Tensor,  # (bs, n_atoms, c_atom)
            atom_pair_repr: Tensor,  # (bs, n_atoms, n_atoms, c_atompair)
            mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Attention mechanism for sequence-local atom attention.
        Args:
            atom_single_repr:
                atom single representation tensor of shape (bs, n_atoms, c_atom)
            atom_single_proj:
                atom single projection tensor of shape (bs, n_atoms, c_atom)
            atom_pair_repr:
                atom pair representation tensor of shape (bs, n_atoms, n_atoms, c_atompair)
            mask:
                atom mask tensor of shape (bs, n_atoms)
        Returns:
            tensor of shape (bs, n_atoms, c_atom) after sequence-local atom attention
        """
        # Input projections
        a = self.ada_ln(atom_single_repr, atom_single_proj)  # AdaLN(a, s)

        # Compute attention pair biases
        pair_bias = self.linear_pair(self.layer_norm_pair(atom_pair_repr))  # (bs, n_atoms, n_atoms, n_heads)

        # Local pair biases (bs, n_atoms // 32, 32, 128, n_heads)
        local_pair_biases = extract_local_biases(pair_bias, self.n_queries, self.n_keys)
        if mask is not None:
            pair_mask = compute_pair_attention_mask(mask).expand(pair_bias.shape)
            local_pair_biases = local_pair_biases + extract_local_biases(pair_mask, self.n_queries, self.n_keys)
        local_pair_biases = local_pair_biases.permute(0, 1, 4, 2, 3)  # move n_heads to third dimension

        # Compute query and key-value tensors
        atom_qx = partition_tensor(a, self.n_queries, self.n_queries)  # (bs, n_atoms // 32, 32, c_atom)
        atom_kvx = partition_tensor(a, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, c_atom)

        # Attention & flatten
        output = self.attention(q_x=atom_qx,
                                kv_x=atom_kvx,
                                biases=[local_pair_biases]).reshape(atom_single_repr.shape)

        # Output projection (from adaLN-Zero)
        output = F.sigmoid(self.output_proj_linear(output)) * output
        return output


class AtomTransformerBlock(nn.Module):
    """AtomTransformerBlock that applies AtomAttentionPairBias and ConditionedTransitionBlock."""

    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            num_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
    ):
        """Initialize the AtomTransformer module.
            Args:
                c_atom:
                    The number of channels for the atom representation.
                num_heads:
                    Number of parallel attention heads. Note that c_atom will be split across no_heads
                    (i.e. each head will have dimension c_atom // no_heads).
                dropout:
                    Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
                n_queries:
                    The size of the atom window. Defaults to 32.
                n_keys:
                    Number of atoms each atom attends to in local sequence space. Defaults to 128.
                c_atom:
                    The number of channels for the atom representation. Defaults to 128.
                c_atompair:
                    The number of channels for the atom pair representation. Defaults to 16.
        """
        super().__init__()
        self.atom_attention = AtomAttentionPairBias(c_atom=c_atom,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    n_queries=n_queries,
                                                    n_keys=n_keys,
                                                    c_atompair=c_atompair)
        self.transition = ConditionedTransitionBlock(c_atom)

    def forward(
            self,
            atom_single_repr: Tensor,
            atom_single_proj: Tensor,
            atom_pair_repr: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        a = self.atom_attention(atom_single_repr, atom_single_proj, atom_pair_repr, mask)
        atom_single_repr = a + self.transition(atom_single_repr, atom_single_proj)
        return atom_single_repr, atom_single_proj, atom_pair_repr


class AtomTransformer(nn.Module):
    """AtomTransformer that applies multiple AtomTransformerBlocks."""

    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            blocks_per_ckpt: int = 1,
            clear_cache_between_blocks: bool = False,
    ):
        """Initialize the AtomTransformer module.
        Args:
            c_atom:
                The number of channels for the atom representation.
            num_blocks:
                Number of blocks.
            num_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.
            blocks_per_ckpt:
                Number of AtomTransformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation

        """
        super().__init__()
        self.c_atom = c_atom
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_pair = c_atompair
        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList(
            [AtomTransformerBlock(c_atom=c_atom,
                                  num_heads=num_heads,
                                  dropout=dropout,
                                  n_queries=n_queries,
                                  n_keys=n_keys,
                                  c_atompair=c_atompair)
             for _ in range(num_blocks)]
        )

    def _prep_blocks(
            self,
            atom_single_repr: Tensor,
            atom_single_proj: Tensor,
            atom_pair_repr: Tensor,
            mask: Optional[Tensor] = None,
    ):
        """Prepare the input tensors for the AtomTransformerBlock."""
        blocks = [
            partial(
                block,
                # atom_single_proj=atom_single_proj,
                # atom_pair_repr=atom_pair_repr,
                mask=mask,
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
            atom_single_repr: Tensor,
            atom_single_proj: Tensor,
            atom_pair_repr: Tensor,
            mask: Optional[Tensor] = None
    ):
        """Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement.
        Args:
            atom_single_repr:
                [bs, n_atoms, c_atom] atom single representation tensor
            atom_single_proj:
                [bs, n_atoms, c_atom] atom single projection tensor of shape
            atom_pair_repr:
                [bs, n_atoms, n_atoms, c_atompair] atom pair representation tensor of shape
            mask:
                [bs, n_atoms] atom mask tensor of shape
        """
        blocks = self._prep_blocks(
            atom_single_repr=atom_single_repr,
            atom_single_proj=atom_single_proj,
            atom_pair_repr=atom_pair_repr,
            mask=mask
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        atom_single_repr, atom_single_proj, atom_pair_repr = checkpoint_blocks(
             blocks,
             args=(atom_single_repr, atom_single_proj, atom_pair_repr),
             blocks_per_ckpt=blocks_per_ckpt,
        )
        # for block in blocks:
        #    atom_single_repr = block(atom_single_repr)
        # for i in range(self.num_blocks):
        #    atom_single_repr = self.blocks[i](atom_single_repr, atom_single_proj, atom_pair_repr, mask)
        return atom_single_repr


def gather_token_repr(
        token_repr: Tensor,  # (bs, n_tokens, c_token)
        tok_idx: Tensor  # (bs, n_atoms)
) -> Tensor:
    """
    Gather token representations based on indices from tok_idx.

    Args:
        token_repr (torch.Tensor):
            Tensor of shape (batch_size, n_tokens, c_token).
        tok_idx (torch.Tensor):
            Tensor of shape (batch_size, n_atoms) containing token indices.

    Returns:
    torch.Tensor: Tensor of shape (batch_size, n_atoms, c_token) with gathered token embeddings.
    """
    batch_size, n_atoms = tok_idx.shape
    _, n_tokens, embed_dim = token_repr.shape
    tok_idx = tok_idx.to(torch.int64)

    # Expand tok_idx to have the same number of dimensions as token_repr
    tok_idx_expanded = tok_idx.unsqueeze(-1).expand(batch_size, n_atoms, embed_dim)

    # Use torch.gather to gather embeddings from token_repr
    gathered_embeddings = torch.gather(token_repr, 1, tok_idx_expanded)

    return gathered_embeddings


def map_token_pairs_to_atom_pairs(
        token_pairs: Tensor,  # (bs, n_tokens, c_pair)
        tok_idx: Tensor  # (bs, n_atoms)
) -> Tensor:
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
    tok_idx = tok_idx.to(torch.int64)

    # Expand tok_idx for efficient gather operation
    tok_idx_l = tok_idx.unsqueeze(2).expand(-1, -1, n_atoms)
    tok_idx_m = tok_idx.unsqueeze(1).expand(-1, n_atoms, -1)
    batch_index = torch.arange(bs, device=token_pairs.device).reshape(bs, 1, 1)

    # Gather token pair embeddings using advanced indexing
    atom_pairs = token_pairs[batch_index, tok_idx_l, tok_idx_m, :]

    return atom_pairs


def aggregate_atom_to_token(
        atom_representation,  # (bs, n_atoms, c_atom)
        tok_idx: Tensor,  # (bs, n_atoms)
        n_tokens: int
) -> Tensor:
    """
    Aggregates atom representations to token representations.

    Args:
        atom_representation (torch.Tensor):
            The atom representations tensor of shape (bs, n_atoms, c_atom).
        tok_idx (torch.Tensor):
            The index tensor of shape (bs, n_atoms) indicating which token each atom belongs to.
        n_tokens (int):
            The number of tokens.
    Returns:
        Aggregated token representations of shape (bs, n_tokens, c_atom).
    Warning: this method is masking aware as long as tok_idx does not encode a mapping like
    masked_atom -> legitimate_token
    """
    bs, n_atoms, c_atom = atom_representation.shape
    tok_idx = tok_idx.to(torch.int64)

    # Initialize the token representation tensor with zeros
    token_representation = torch.zeros((bs, n_tokens, c_atom),
                                       device=atom_representation.device,
                                       dtype=atom_representation.dtype)

    # Expand tok_idx to make it compatible for scattering with atom_representation
    tok_idx_expanded = tok_idx.unsqueeze(-1).expand(-1, -1, c_atom)  # (bs, n_atoms, c_atom)

    # Aggregate atom representations into token representations
    token_representation = token_representation.scatter_reduce(dim=1,
                                                               index=tok_idx_expanded,
                                                               src=atom_representation,
                                                               reduce='mean',
                                                               include_self=False)
    return token_representation


class AtomAttentionEncoderOutput(NamedTuple):
    """Structured output class for AtomAttentionEncoder."""
    token_single: torch.Tensor  # (bs, n_tokens, c_token)
    atom_single_skip_repr: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_single_skip_proj: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_pair_skip_repr: torch.Tensor  # (bs, n_atoms, n_atoms c_atompair)


class AtomAttentionEncoder(nn.Module):
    """AtomAttentionEncoder"""

    def __init__(
            self,
            n_tokens: int,
            c_token: int,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 4,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            trunk_conditioning: bool = False,
            clear_cache_between_blocks: bool = False,
    ):
        """Initialize the AtomAttentionEncoder module.
        Args:
            n_tokens:
                The number of tokens that will be in the output representation.
            c_token:
                The number of channels for the token representation.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_atompair:
                The number of channels for the pair representation. Defaults to 16.
            c_trunk_pair:
                The number of channels for the trunk pair representation. Defaults to 16.
            num_blocks:
                Number of blocks in AtomTransformer. Defaults to 3.
            num_heads:
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
        self.n_tokens = n_tokens
        self.num_blocks = num_blocks
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_trunk_pair = c_trunk_pair
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.trunk_conditioning = trunk_conditioning
        self.clear_cache_between_blocks = clear_cache_between_blocks

        # Embedding per-atom metadata, concat(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)
        self.linear_atom_embedding = LinearNoBias(3 + 1 + 1 + 128 + 4 * 64, c_atom)

        # Embedding offsets between atom reference positions
        self.linear_atom_offsets = LinearNoBias(3, c_atompair)
        self.linear_atom_distances = LinearNoBias(1, c_atompair)

        # Embedding the valid mask
        self.linear_mask = LinearNoBias(1, c_atompair)

        if trunk_conditioning:
            self.linear_trunk_single = LinearNoBias(c_token, c_atom, init='final')
            self.trunk_single_layer_norm = nn.LayerNorm(c_token)

            self.trunk_pair_layer_norm = nn.LayerNorm(c_trunk_pair)
            self.linear_trunk_pair = LinearNoBias(c_trunk_pair, c_atompair, init='final')

            self.linear_noisy_pos = LinearNoBias(3, c_atom, init='final')

        # Adding the single conditioning to the pair representation
        self.linear_single_to_pair_row = LinearNoBias(c_atom, c_atompair, init='relu')
        self.linear_single_to_pair_col = LinearNoBias(c_atom, c_atompair, init='relu')

        # Small MLP on the pair activations
        self.linear_pair_1 = LinearNoBias(c_atompair, c_atompair, init='relu')
        self.linear_pair_2 = LinearNoBias(c_atompair, c_atompair, init='final')

        # Cross attention transformer
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        # Final linear
        self.linear_output = LinearNoBias(c_atom, c_token, init='relu')

    def _prep_pair_repr(
            self,
            features: Dict[str, Tensor],
            atom_single: Tensor,
            z_trunk: Optional[Tensor],
    ) -> Tensor:
        """Compute the pair representation for the atom transformer.
        This is done in a separate function for checkpointing. The intermediate activations due to the
        atom pair representations are large and can be checkpointed to reduce memory usage.
        Args:
            features:
                Dictionary of input features.
            atom_single:
                [*, n_atoms, c_atom] The single atom representation from _prep_single_repr
            z_trunk:
                [*, n_tokens, n_tokens, c_trunk] the pair representation from the trunk
        """
        # Embed offsets between atom reference positions
        offsets = features['ref_pos'][:, :, None, :] - features['ref_pos'][:, None, :, :]  # (bs, n_atoms, n_atoms, 3)
        valid_mask = features['ref_space_uid'][:, :, None] == features['ref_space_uid'][:, None, :]
        valid_mask = valid_mask.unsqueeze(-1).to(offsets.dtype)  # convert boolean to binary
        atom_pair = self.linear_atom_offsets(offsets) * valid_mask

        # Embed pairwise inverse squared distances, and the valid mask
        squared_distances = offsets.pow(2).sum(dim=-1, keepdim=True)  # (bs, n_atoms, n_atoms, 1)
        inverse_dists = torch.reciprocal(torch.add(squared_distances, 1))
        atom_pair = atom_pair + self.linear_atom_distances(inverse_dists) * valid_mask
        atom_pair = atom_pair + self.linear_mask(valid_mask) * valid_mask

        # If provided, add trunk embeddings
        if self.trunk_conditioning:
            atom_pair = atom_pair + map_token_pairs_to_atom_pairs(
                self.linear_trunk_pair(self.trunk_pair_layer_norm(z_trunk)),
                features['atom_to_token']
            )

        # Add the combined single conditioning to the pair representation
        atom_pair = self.linear_single_to_pair_row(F.relu(atom_single[:, None, :, :])) + \
                    self.linear_single_to_pair_col(F.relu(atom_single[:, :, None, :])) + atom_pair

        # Run a small MLP on the pair activations
        atom_pair = self.linear_pair_2(F.relu(self.linear_pair_1(F.relu(atom_pair))))

        return atom_pair

    def _prep_single_repr(
            self,
            features: Dict[str, Tensor],
            s_trunk: Optional[Tensor],
            noisy_pos: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Compute the single representation for the atom transformer.
        This is done in a separate function for checkpointing. The intermediate activations due to the
        atom single representations are large and can be checkpointed to reduce memory usage.
        Args:
            features:
                Dictionary of input features.
            s_trunk:
                [*, n_tokens, c_token] the token representation from the trunk
            noisy_pos:
                [*, n_atoms, 3] the noisy atom positions
        """
        batch_size, n_atoms, _ = features['ref_pos'].size()

        # Embed atom metadata
        atom_single_conditioning = self.linear_atom_embedding(
            torch.cat(
                [features['ref_pos'],
                 features['ref_charge'].unsqueeze(-1),
                 features['ref_mask'].unsqueeze(-1),
                 features['ref_element'],
                 features['ref_atom_name_chars'].reshape(batch_size, n_atoms, 4 * 64)],
                dim=2
            )
        )
        # Initialize the atom single representation as the single conditioning
        atom_single = atom_single_conditioning.clone()
        # atom_single_conditioning -> c_l in AF3 Supplement
        # atom_single -> q_l in AF3 Supplement

        # If provided, add trunk embeddings and noisy positions
        if self.trunk_conditioning:
            atom_single_conditioning = atom_single_conditioning + gather_token_repr(
                self.linear_trunk_single(self.trunk_single_layer_norm(s_trunk)),
                features['atom_to_token']
            )

            # Add the noisy positions
            atom_single = atom_single + self.linear_noisy_pos(noisy_pos)

        return atom_single, atom_single_conditioning

    def forward(
            self,
            features: Dict[str, Tensor],
            s_trunk: Optional[Tensor] = None,  # (bs, n_tokens, c_token)
            z_trunk: Optional[Tensor] = None,  # (bs, n_tokens, c_trunk_pair)
            noisy_pos: Optional[Tensor] = None,  # (bs, n_atoms, 3)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
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
            s_trunk:
                [*, N_tokens, c_token] single representation of the Pairformer trunk
            z_trunk:
                [*, N_tokens, N_tokens, c_pair] pair representation of the Pairformer trunk
            noisy_pos:
                [*, N_atoms, 3] Tensor containing the noisy positions. Defaults to None.
            mask:
                [*, N_atoms]
        Returns:
            A named tuple containing the following fields:
                token_single:
                    [*, N_tokens, c_token] single representation
                atom_single_skip_repr:
                    [*, N_atoms, c_atom] atom single representation (denoted q_l in AF3 Supplement)
                atom_single_skip_proj:
                    [*, N_atoms, c_atom] atom single projection (denoted c_l in AF3 Supplement)
                atom_pair_skip_repr:
                    [*, N_atoms, N_atoms, c_atompair] atom pair representation (denoted p_lm in AF3 Supplement)
        """
        # batch_size, n_atoms, _ = features['ref_pos'].size()
        # Create the atom single conditioning: Embed per-atom metadata
        # atom_single = self.linear_atom_embedding(
        #    torch.cat(
        #        [features['ref_pos'],
        #         features['ref_charge'].unsqueeze(-1),
        #         features['ref_mask'].unsqueeze(-1),
        #         features['ref_element'],
        #         features['ref_atom_name_chars'].reshape(batch_size, n_atoms, 4 * 64)],
        #        dim=2
        #    )
        # )

        # Initialize the atom single representation as the single conditioning
        # atom_single_conditioning = torch.clone(atom_single)  # (bs, n_atoms, c_atom)
        # atom_single_conditioning -> q_l in AF3 Supplement
        # atom_single -> c_l in AF3 Supplement

        # If provided, add trunk embeddings and noisy positions
        # if self.trunk_conditioning:
        #    atom_single = atom_single + gather_token_repr(
        #        self.linear_trunk_single(self.trunk_single_layer_norm(s_trunk)),
        #        features['atom_to_token']
        #    )

        # Add the noisy positions
        #    atom_single_conditioning = atom_single_conditioning + self.linear_noisy_pos(noisy_pos)

        # Initialize representations
        atom_single, atom_single_conditioning = checkpoint(self._prep_single_repr, features, s_trunk, noisy_pos)
        atom_pair = checkpoint(self._prep_pair_repr, features, atom_single, z_trunk)

        # Cross attention transformer
        atom_single_conditioning = self.atom_transformer(atom_single_conditioning, atom_single, atom_pair, mask)

        # Aggregate per-atom representation to per-token representation
        token_repr = aggregate_atom_to_token(atom_representation=F.relu(self.linear_output(atom_single_conditioning)),
                                             tok_idx=features['atom_to_token'],
                                             n_tokens=self.n_tokens)
        output = AtomAttentionEncoderOutput(
            token_single=token_repr,
            atom_single_skip_repr=atom_single_conditioning,
            atom_single_skip_proj=atom_single,
            atom_pair_skip_repr=atom_pair,
        )
        return output


class AtomAttentionDecoder(nn.Module):
    """AtomAttentionDecoder that broadcasts per-token activations to per-atom activations."""

    def __init__(
            self,
            c_token: int,
            c_atom: int = 128,
            c_atompair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            clear_cache_between_blocks: bool = False,
    ):
        """Initialize the AtomAttentionDecoder module.
        Args:
            c_token:
                The number of channels for the token representation.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_atompair:
                The number of channels for the atom pair representation. Defaults to 16.
            num_blocks:
                Number of blocks.
            num_heads:
                Number of parallel attention heads. Note that c_atom will be split across no_heads
                (i.e. each head will have dimension c_atom // no_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation

        """
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        self.linear_atom = LinearNoBias(c_token, c_atom, init='default')
        self.linear_update = LinearNoBias(c_atom, 3, init='final')
        self.layer_norm = nn.LayerNorm(c_atom)

    def forward(
            self,
            token_repr: Tensor,  # (bs, n_tokens, c_token)
            atom_single_skip_repr: Tensor,  # (bs, n_atoms, c_atom)
            atom_single_skip_proj: Tensor,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr: Tensor,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx: Tensor,  # (bs, n_atoms)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
    ) -> Tensor:
        """AtomAttentionDecoder. Algorithm 6 in AlphaFold3 supplement.
        Args:
            token_repr:
                [bs, n_tokens, c_atom] Per-token activations.
            atom_single_skip_repr:
                [bs, n_atoms, c_atom] Per-atom activations added as the skip connection.
            atom_single_skip_proj:
                [bs, n_atoms, c_atom] Per-atom activations provided to AtomTransformer.
            atom_pair_skip_repr:
                [bs, n_atoms, n_atoms, c_atom] Pair activations provided to AtomTransformer.
            tok_idx:
                [bs, n_atoms] Token indices that encode which token each atom belongs to.
            mask:
                [bs, n_atoms] Mask for the atom transformer.
        Returns:
            [bs, n_atoms, 3] a tensor of per-atom coordinate updates.
        """
        # Broadcast per-token activations to per-atom activations and add the skip connection
        atom_single_repr = self.linear_atom(gather_token_repr(token_repr, tok_idx)) + atom_single_skip_repr

        # Cross-attention transformer
        atom_single_repr = self.atom_transformer(atom_single_repr, atom_single_skip_proj, atom_pair_skip_repr, mask)

        # Map to positions update
        r_atom_update = self.linear_update(self.layer_norm(atom_single_repr))
        return r_atom_update


"""
def _split_heads(x, n_heads):
    ""Split the last dimension of a tensor into multiple heads.""
    # x has shape (batch_size, seq_length, 128, c_atom)
    batch_size, seq_length, tokens, embed_dim = x.shape

    # Validate that c_atom can be divided by n_heads
    if embed_dim % n_heads != 0:
        raise ValueError("c_atom must be divisible by n_heads")

    # Reshape
    new_shape = (batch_size, seq_length, tokens, n_heads, embed_dim // n_heads)
    x = x.reshape(new_shape)

    # Permute to get (batch_size, n_heads, seq_length, tokens, feature_dim)
    x = x.permute(0, 3, 1, 2, 4)  # move n_heads to the second position
    return x

def _concatenate_heads(x):
    ""Concatenate the heads in the second dimension of a tensor along the final dimension.""
    # x has shape (bs, n_heads, n_atoms // 32, 32, c_atom // n_heads)
    bs, n_heads, seq_length, tokens, head_dim = x.shape

    # Permute to bring heads to the last dimension before combining
    x = x.permute(0, 2, 3, 1, 4)  # shape becomes (bs, n_atoms // 32, 32, n_heads, c_atom // n_heads)

    # Reshape to concatenate the head dimensions
    new_shape = (bs, seq_length, tokens, n_heads * head_dim)
    x = x.reshape(new_shape)
    return x
"""
