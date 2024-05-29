"""Sequence-local atom attention.
The 'sequence-local atom attention' represents the whole structure as a flat list of atoms and allows all atoms to
'talk' directly to each other within a certain sequence neighbourhood. e.g. each subset of 32 atoms attends to the
subset of the nearby 128 atoms (nearby in the sequence space). This gives the network the capacity to learn general
rules about local atom constellations, independently of the coarse-grained tokenization where each standard residue
is represented with a single token only."""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from src.models.components.primitives import AdaLN, Linear
from src.models.components.transition import ConditionedTransitionBlock
from src.utils.tensor_utils import partition_tensor
from typing import Dict, Tuple


def _split_heads(x, n_heads):
    """Split the last dimension of a tensor into multiple heads."""
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
    """Concatenate the heads in the second dimension of a tensor along the final dimension."""
    # x has shape (bs, n_heads, n_atoms // 32, 32, c_atom // n_heads)
    bs, n_heads, seq_length, tokens, head_dim = x.shape

    # Permute to bring heads to the last dimension before combining
    x = x.permute(0, 2, 3, 1, 4)  # shape becomes (bs, n_atoms // 32, 32, n_heads, c_atom // n_heads)

    # Reshape to concatenate the head dimensions
    new_shape = (bs, seq_length, tokens, n_heads * head_dim)
    x = x.reshape(new_shape)
    return x


def extract_local_biases(bias_tensor, partition_increment=32, partition_length=128):
    """Extracts biases that are local in the sequence space
    Args:
        bias_tensor:
            A tensor of shape [batch_size, N_atoms, N_atoms, channels].
        partition_increment:
            The increment between the centers of the partitions.
        partition_length:
            The length of the partitions.
    Returns:
        A tensor of shape [batch_size, N_atoms // partition_increment, partition_length, channels].
    """
    batch_size, N_atoms, N_atoms, channels = bias_tensor.shape
    half_length = partition_length // 2
    assert N_atoms > partition_length, "Number of atoms must be greater than partition length."
    assert N_atoms % partition_increment == 0, "Number of atoms must be divisible by partition increment."

    # Calculate centers starting from 15.5 with an increment of 32
    centers = np.arange(partition_increment / 2, N_atoms, step=partition_increment, dtype='float32')

    # Initialize a list to hold the partitions
    partitions = []

    for i, center in enumerate(centers):
        # Calculate start and end indices
        start_column_index = int(center - half_length)
        end_column_index = int(center + half_length)
        start_row_index = i * partition_increment
        end_row_index = (i + 1) * partition_increment

        # Apply padding if necessary and extract the slice
        if start_column_index < 0:
            # Pad at the beginning
            pre_padding = torch.zeros((batch_size, partition_increment, -start_column_index, channels),
                                      device=bias_tensor.device)
            valid_part = bias_tensor[:, start_row_index:end_row_index, :end_column_index, :]
            partition = torch.cat([pre_padding, valid_part], dim=2)
        elif end_column_index > N_atoms:
            # Pad at the end
            post_padding = torch.zeros((batch_size, partition_increment, end_column_index - N_atoms, channels),
                                       device=bias_tensor.device)
            valid_part = bias_tensor[:, start_row_index:end_row_index, start_column_index:N_atoms, :]
            partition = torch.cat([valid_part, post_padding], dim=2)
        else:
            # No padding needed
            partition = bias_tensor[:, start_row_index:end_row_index, start_column_index:end_column_index, :]
        partitions.append(partition)

    # Stack all the partitions along a new dimension
    output_tensor = torch.stack(partitions, dim=1)

    return output_tensor


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
            device=None,
            dtype=None,
    ):
        """Initialize the AtomAttentionPairBias module.
        Args:
            c_atom:
                Total dimension of the model.
            num_heads:
                Number of parallel attention heads. Note that c_atom will be split across num_heads
                (i.e. each head will have dimension c_atom // num_heads).
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
        self.device = device
        self.dtype = dtype
        self.n_queries = n_queries
        self.n_keys = n_keys

        # Projections
        self.ada_ln = AdaLN(c_atom)
        self.output_proj_linear = Linear(c_atom, c_atom, init='gating')
        self.output_proj_linear.bias = nn.Parameter(torch.ones(c_atom) * -2.0)  # gate values will be ~0.11

        # QKV projections
        self.q_linear = Linear(c_atom, c_atom, init='glorot')
        self.k_linear = Linear(c_atom, c_atom, init='glorot', bias=False)
        self.v_linear = Linear(c_atom, c_atom, init='glorot', bias=False)

        # Pair bias
        self.layer_norm_pair = nn.LayerNorm(self.c_atompair)
        self.linear_pair = Linear(self.c_atompair, self.num_heads, init='default', bias=False)

        # Gating
        self.gating_linear = Linear(c_atom, c_atom, init='gating', bias=False)
        self.attention_proj = Linear(c_atom, c_atom, init='default', bias=False)

    def forward(self, atom_single_repr, atom_single_proj, atom_pair_repr, mask=None):
        """
        Attention mechanism for sequence-local atom attention.
        Args:
            atom_single_repr:
                tensor of shape (bs, n_atoms, c_atom)
            atom_single_proj:
                tensor of shape (bs, n_atoms, c_atom)
            atom_pair_repr:
                tensor of shape (bs, n_atoms, n_atoms, c_atompair)
            mask:
                tensor of shape (bs, n_atoms)
        Returns:
            tensor of shape (bs, n_atoms, c_atom) after sequence-local atom attention
        TODO: implement masking
        """
        # Input projections
        a = self.ada_ln(atom_single_repr, atom_single_proj)  # AdaLN(a, s)

        # Project query, key and value vectors
        q = self.q_linear(a)  # (bs, n_atoms, c_atom)
        k = self.k_linear(a)
        v = self.v_linear(a)

        # Sequence-local atom attention
        q = partition_tensor(q, self.n_queries, self.n_queries)  # (bs, n_atoms // 32, 32, c_atom)
        k = partition_tensor(k, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, c_atom)
        v = partition_tensor(v, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, c_atom)

        # Split heads and rearrange
        q = _split_heads(q, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, c_atom // n_heads)
        k = _split_heads(k, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, c_atom // n_heads)
        v = _split_heads(v, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, c_atom // n_heads)

        # Compute attention pair biases
        pair_bias = self.linear_pair(self.layer_norm_pair(atom_pair_repr))  # (bs, n_atoms, n_atoms, n_heads)

        # Local pair biases (bs, n_atoms // 32, 32, 128, n_heads)
        local_pair_biases = extract_local_biases(pair_bias, self.n_queries, self.n_keys)
        local_pair_biases = local_pair_biases.permute(0, 4, 1, 2, 3)  # move n_heads to second dimension

        # Attention  (bs, n_heads, n_atoms // 32, 32, c_atom // n_heads)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=local_pair_biases, dropout_p=self.dropout)
        attention_output = _concatenate_heads(attention_output).reshape(atom_single_repr.shape)  # concat and flatten

        # Gating
        gated_output = F.sigmoid(self.gating_linear(attention_output)) * attention_output
        output = self.attention_proj(gated_output)  # (bs, n_atoms, c_atom)

        # Output projection (from adaLN-Zero)
        output = F.sigmoid(self.output_proj_linear(output)) * output

        return output


class AtomTransformer(nn.Module):
    """AtomTransformer that applies multiple blocks of AttentionPairBias and ConditionedTransitionBlock."""

    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            device=None,
            dtype=None,
    ):
        """Initialize the AtomTransformer module.
        Args:
            c_atom:
                The number of channels for the atom representation.
            num_blocks:
                Number of blocks.
            num_heads:
                Number of parallel attention heads. Note that c_atom will be split across num_heads
                (i.e. each head will have dimension c_atom // num_heads).
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
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_pair = c_atompair
        self.device = device
        self.dtype = dtype

        self.attention_blocks = nn.ModuleList(
            [AtomAttentionPairBias(c_atom=c_atom,
                                   num_heads=num_heads,
                                   dropout=dropout,
                                   n_queries=n_queries,
                                   n_keys=n_keys,
                                   c_atompair=c_atompair,
                                   device=device,
                                   dtype=dtype)
             for _ in range(num_blocks)]
        )
        self.conditioned_transition_blocks = nn.ModuleList(
            [ConditionedTransitionBlock(c_atom) for _ in range(num_blocks)]
        )

    def forward(self, atom_single_repr, atom_single_proj, atom_pair_repr, mask=None):
        """Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement."""
        for i in range(self.num_blocks):
            b = self.attention_blocks[i](atom_single_repr, atom_single_proj, atom_pair_repr, mask)
            atom_single_repr = b + self.conditioned_transition_blocks[i](atom_single_repr, atom_single_proj)
        return atom_single_repr


def gather_token_repr(token_repr, tok_idx):
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

    # Expand tok_idx to have the same number of dimensions as token_repr
    tok_idx_expanded = tok_idx.unsqueeze(-1).expand(batch_size, n_atoms, embed_dim)

    # Use torch.gather to gather embeddings from token_repr
    gathered_embeddings = torch.gather(token_repr, 1, tok_idx_expanded)

    return gathered_embeddings


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


def aggregate_atom_to_token(
        atom_representation,  # (bs, n_atoms, c_atom)
        tok_idx: torch.Tensor,  # (bs, n_atoms)
        n_tokens: int
) -> torch.Tensor:
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
    """
    bs, n_atoms, c_atom = atom_representation.shape
    # Initialize the token representation tensor with zeros
    token_representation = torch.zeros(bs, n_tokens, c_atom,
                                       device=atom_representation.device,
                                       dtype=atom_representation.dtype)

    # Expand tok_idx to make it compatible for scattering with atom_representation
    tok_idx_expanded = tok_idx.unsqueeze(-1).expand(-1, -1, c_atom)  # (bs, n_atoms, c_atom)

    # Aggregate atom representations into token representations using scatter_reduce
    token_representation = token_representation.scatter_reduce_(dim=1,
                                                                index=tok_idx_expanded,
                                                                src=atom_representation,
                                                                reduce='mean',
                                                                include_self=False)
    return token_representation


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
            device=None,
            dtype=None,
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
                Number of parallel attention heads. Note that c_atom will be split across num_heads
                (i.e. each head will have dimension c_atom // num_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.

            trunk_conditioning:
                Whether to condition the atom single and atom-pair representation on the trunk.
                Defaults to False.

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
        self.device = device
        self.dtype = dtype

        # Embedding per-atom metadata, concat(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars)
        self.linear_atom_embedding = Linear(3 + 1 + 1 + 128 + 4 * 64, c_atom, bias=False)

        # Embedding offsets between atom reference positions
        self.linear_atom_offsets = Linear(3, c_atompair, bias=False)
        self.linear_atom_distances = Linear(1, c_atompair, bias=False)

        # Embedding the valid mask
        self.linear_mask = Linear(1, c_atompair, bias=False)

        if trunk_conditioning:
            self.linear_trunk_single = Linear(c_token, c_atom, bias=False, init='final')
            self.trunk_single_layer_norm = nn.LayerNorm(c_token)

            self.trunk_pair_layer_norm = nn.LayerNorm(c_trunk_pair)
            self.linear_trunk_pair = Linear(c_trunk_pair, c_atompair, bias=False, init='final')

            self.linear_noisy_pos = Linear(3, c_atom, bias=False, init='final')

        # Adding the single conditioning to the pair representation
        self.linear_single_to_pair_row = Linear(c_atom, c_atompair, bias=False, init='relu')
        self.linear_single_to_pair_col = Linear(c_atom, c_atompair, bias=False, init='relu')

        # Small MLP on the pair activations
        self.linear_pair_1 = Linear(c_atompair, c_atompair, bias=False, init='relu')
        self.linear_pair_2 = Linear(c_atompair, c_atompair, bias=False, init='final')

        # Cross attention transformer
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            device=device,
            dtype=dtype
        )

        # Final linear
        self.linear_output = Linear(c_atom, c_token, bias=False, init='relu')

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            pairformer_output: Dict[str, torch.tensor] = None,
            noisy_pos: torch.Tensor = None,  # (bs, n_atoms, 3)
    ) -> Dict[str, torch.Tensor]:
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
                    "tok_idx":
                        [*, N_atoms] Token index for each atom in the flat atom representation.
            pairformer_output:
                Dictionary containing the output of the Pairformer model:
                    "token_single":
                        [*, N_tokens, c_token] single representation
                    "token_pair":
                        [*, N_tokens, N_tokens, c_pair] pair representation
            noisy_pos:
                [*, N_atoms, 3] Tensor containing the noisy positions. Defaults to None.
        Returns:
            Dictionary containing the output features:
                "token_single":
                    [*, N_tokens, c_token] single representation
                "atom_single_skip_repr":
                    [*, N_atoms, c_atom] atom single representation (denoted q_l in AF3 Supplement)
                "atom_single_skip_proj":
                    [*, N_atoms, c_atom] atom single projection (denoted c_l in AF3 Supplement)
                "atom_pair_skip_repr":
                    [*, N_atoms, N_atoms, c_atompair] atom pair representation (denoted p_lm in AF3 Supplement)
        """
        batch_size, n_atoms, _ = features['ref_pos'].size()
        # Create the atom single conditioning: Embed per-atom metadata
        atom_single = self.linear_atom_embedding(
            torch.cat(
                [features['ref_pos'],
                 features['ref_charge'].unsqueeze(-1),
                 features['ref_mask'].unsqueeze(-1),
                 features['ref_element'],
                 features['ref_atom_name_chars'].reshape(batch_size, n_atoms, 4 * 64)],
                dim=2
            )
        )

        # Embed offsets between atom reference positions
        offsets = features['ref_pos'][:, :, None, :] - features['ref_pos'][:, None, :, :]  # (bs, n_atoms, n_atoms, 3)
        valid_mask = features['ref_mask'][:, :, None] == features['ref_mask'][:, None, :]  # (bs, n_atoms, n_atoms)
        valid_mask = valid_mask.unsqueeze(-1).float()  # convert boolean to binary where 1.0 is True, 0.0 is False
        atom_pair = self.linear_atom_offsets(offsets) * valid_mask

        # Embed pairwise inverse squared distances, and the valid mask
        squared_distances = offsets.pow(2).sum(dim=-1, keepdim=True)  # (bs, n_atoms, n_atoms, 1)
        inverse_dists = torch.reciprocal(torch.add(squared_distances, 1))
        atom_pair = atom_pair + self.linear_atom_distances(inverse_dists) * valid_mask
        atom_pair = atom_pair + self.linear_mask(valid_mask) * valid_mask

        # Initialize the atom single representation as the single conditioning
        atom_single_conditioning = torch.clone(atom_single)  # (bs, n_atoms, c_atom)
        # atom_single_conditioning -> q_l in AF3 Supplement
        # atom_single -> c_l in AF3 Supplement

        # If provided, add trunk embeddings and noisy positions
        if self.trunk_conditioning:
            atom_single = atom_single + self.linear_trunk_single(
                self.trunk_single_layer_norm(gather_token_repr(pairformer_output['token_single'], features['tok_idx']))
            )
            atom_pair = atom_pair + self.linear_trunk_pair(
                self.trunk_pair_layer_norm(map_token_pairs_to_atom_pairs(
                    pairformer_output['token_pair'], features['tok_idx'])
                )
            )
            # Add the noisy positions
            atom_single_conditioning = atom_single_conditioning + self.linear_noisy_pos(noisy_pos)

        # Add the combined single conditioning to the pair representation
        atom_pair = self.linear_single_to_pair_row(F.relu(atom_single[:, None, :, :])) + \
                    self.linear_single_to_pair_col(F.relu(atom_single[:, :, None, :])) + atom_pair

        # Run a small MLP on the pair activations
        atom_pair = self.linear_pair_2(F.relu(self.linear_pair_1(F.relu(atom_pair))))

        # Cross attention transformer
        atom_single_conditioning = self.atom_transformer(atom_single_conditioning, atom_single, atom_pair)

        # Aggregate per-atom representation to per-token representation
        token_repr = aggregate_atom_to_token(atom_representation=F.relu(self.linear_output(atom_single_conditioning)),
                                             tok_idx=features['tok_idx'],
                                             n_tokens=self.n_tokens)
        output_dict = {
            "token_single": token_repr,
            "atom_single_skip_repr": atom_single_conditioning,
            "atom_single_skip_proj": atom_single,
            "atom_pair_skip_repr": atom_pair,
        }
        return output_dict


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
            device=None,
            dtype=None,
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
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.device = device
        self.dtype = dtype

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            device=device,
            dtype=dtype
        )

        self.linear_atom = Linear(c_token, c_atom, init='default', bias=False)
        self.linear_update = Linear(c_atom, c_atom, init='default', bias=False)
        self.layer_norm = nn.LayerNorm(c_atom)

    def forward(
            self,
            token_repr,  # (bs, n_tokens, c_token)
            atom_single_skip_repr,  # (bs, n_atoms, c_atom)
            atom_single_skip_proj,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx,  # (bs, n_atoms)
            mask=None,  # (bs, n_atoms)
    ):
        """AtomAttentionDecoder. Algorithm 6 in AlphaFold3 supplement.
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
        """
        # Broadcast per-token activations to per-atom activations and add the skip connection
        atom_single_repr = self.linear_atom(gather_token_repr(token_repr, tok_idx)) + atom_single_skip_repr

        # Cross-attention transformer
        atom_single_repr = self.atom_transformer(atom_single_repr, atom_single_skip_proj, atom_pair_skip_repr, mask)

        # Map to positions update
        r_atom_update = self.linear_update(self.layer_norm(atom_single_repr))
        return r_atom_update
