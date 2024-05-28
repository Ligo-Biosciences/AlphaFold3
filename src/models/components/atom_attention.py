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


def _split_heads(x, n_heads):
    """Split the last dimension of a tensor into multiple heads."""
    # x has shape (batch_size, seq_length, 128, embed_dim)
    batch_size, seq_length, tokens, embed_dim = x.shape

    # Validate that embed_dim can be divided by n_heads
    if embed_dim % n_heads != 0:
        raise ValueError("embed_dim must be divisible by n_heads")

    # Reshape
    new_shape = (batch_size, seq_length, tokens, n_heads, embed_dim // n_heads)
    x = x.reshape(new_shape)

    # Permute to get (batch_size, n_heads, seq_length, tokens, feature_dim)
    x = x.permute(0, 3, 1, 2, 4)  # move n_heads to the second position
    return x


def _concatenate_heads(x):
    """Concatenate the heads in the second dimension of a tensor along the final dimension."""
    # x has shape (bs, n_heads, n_atoms // 32, 32, embed_dim // n_heads)
    bs, n_heads, seq_length, tokens, head_dim = x.shape

    # Permute to bring heads to the last dimension before combining
    x = x.permute(0, 2, 3, 1, 4)  # shape becomes (bs, n_atoms // 32, 32, n_heads, embed_dim // n_heads)

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
            embed_dim,
            num_heads=8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            c_atom: int = 128,
            c_pair: int = 16,
            device=None,
            dtype=None,
    ):
        """Initialize the AtomAttentionPairBias module.
        Args:
            embed_dim:
                Total dimension of the model.
            num_heads:
                Number of parallel attention heads. Note that embed_dim will be split across num_heads
                (i.e. each head will have dimension embed_dim // num_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_pair = c_pair

        # Projections
        self.ada_ln = AdaLN(embed_dim)
        self.output_proj_linear = Linear(embed_dim, embed_dim, init='gating')
        self.output_proj_linear.bias = nn.Parameter(torch.ones(embed_dim) * -2.0)  # gate values will be ~0.11

        # QKV projections
        self.q_linear = Linear(embed_dim, embed_dim, init='glorot')
        self.k_linear = Linear(embed_dim, embed_dim, init='glorot', bias=False)
        self.v_linear = Linear(embed_dim, embed_dim, init='glorot', bias=False)

        # Pair bias
        self.layer_norm_pair = nn.LayerNorm(self.c_pair)
        self.linear_pair = Linear(self.c_pair, self.num_heads, init='default', bias=False)

        # Gating
        self.gating_linear = Linear(embed_dim, embed_dim, init='gating', bias=False)
        self.attention_proj = Linear(embed_dim, embed_dim, init='default', bias=False)

    def forward(self, atom_single_repr, atom_single_proj, atom_pair_repr, mask=None):
        """
        Attention mechanism for sequence-local atom attention.
        Args:
            atom_single_repr:
                tensor of shape (bs, n_atoms, embed_dim)
            atom_single_proj:
                tensor of shape (bs, n_atoms, embed_dim)
            atom_pair_repr:
                tensor of shape (bs, n_atoms, n_atoms, c_pair)
            mask:
                tensor of shape (bs, n_atoms)
        Returns:
            tensor of shape (bs, n_atoms, embed_dim) after sequence-local atom attention
        TODO: implement masking
        """
        # Input projections
        a = self.ada_ln(atom_single_repr, atom_single_proj)  # AdaLN(a, s)

        # Project query, key and value vectors
        q = self.q_linear(a)  # (bs, n_atoms, embed_dim)
        k = self.k_linear(a)
        v = self.v_linear(a)

        # Sequence-local atom attention
        q = partition_tensor(q, self.n_queries, self.n_queries)  # (bs, n_atoms // 32, 32, embed_dim)
        k = partition_tensor(k, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, embed_dim)
        v = partition_tensor(v, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, embed_dim)

        # Split heads and rearrange
        q = _split_heads(q, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, embed_dim // n_heads)
        k = _split_heads(k, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, embed_dim // n_heads)
        v = _split_heads(v, self.num_heads)  # (bs, n_heads, n_atoms // 32, 128, embed_dim // n_heads)

        # Compute attention pair biases
        pair_bias = self.linear_pair(self.layer_norm_pair(atom_pair_repr))  # (bs, n_atoms, n_atoms, n_heads)

        # Local pair biases (bs, n_atoms // 32, 32, 128, n_heads)
        local_pair_biases = extract_local_biases(pair_bias, self.n_queries, self.n_keys)
        local_pair_biases = local_pair_biases.permute(0, 4, 1, 2, 3)  # move n_heads to second dimension

        # Attention  (bs, n_heads, n_atoms // 32, 32, embed_dim // n_heads)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=local_pair_biases, dropout_p=self.dropout)
        attention_output = _concatenate_heads(attention_output).reshape(atom_single_repr.shape)  # concat and flatten

        # Gating
        gated_output = F.sigmoid(self.gating_linear(attention_output)) * attention_output
        output = self.attention_proj(gated_output)  # (bs, n_atoms, embed_dim)

        # Output projection (from adaLN-Zero)
        output = F.sigmoid(self.output_proj_linear(output)) * output

        return output


class AtomTransformer(nn.Module):
    """AtomTransformer that applies multiple blocks of AttentionPairBias and ConditionedTransitionBlock."""
    def __init__(
            self,
            embed_dim: int,
            num_blocks: int,
            num_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            c_atom: int = 128,
            c_pair: int = 16,
            device=None,
            dtype=None,
    ):
        """Initialize the AtomTransformer module.
        Args:
            embed_dim:
                Total dimension of the model.
            num_blocks:
                Number of blocks.
            num_heads:
                Number of parallel attention heads. Note that embed_dim will be split across num_heads
                (i.e. each head will have dimension embed_dim // num_heads).
            dropout:
                Dropout probability on attn_output_weights. Default: 0.0 (no dropout).
            n_queries:
                The size of the atom window. Defaults to 32.
            n_keys:
                Number of atoms each atom attends to in local sequence space. Defaults to 128.
            c_atom:
                The number of channels for the atom representation. Defaults to 128.
            c_pair:
                The number of channels for the pair representation. Defaults to 16.

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_pair = c_pair
        self.device = device
        self.dtype = dtype

        self.attention_blocks = nn.ModuleList(
            [AtomAttentionPairBias(embed_dim, num_heads, dropout, n_queries, n_keys, c_atom, c_pair, device, dtype)
             for _ in range(num_blocks)]
        )
        self.conditioned_transition_blocks = nn.ModuleList(
            [ConditionedTransitionBlock(embed_dim) for _ in range(num_blocks)]
        )

    def forward(self, atom_single_repr, atom_single_proj, atom_pair_repr, mask=None):
        """Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement."""
        for i in range(self.num_blocks):
            b = self.attention_blocks[i](atom_single_repr, atom_single_proj, atom_pair_repr, mask)
            atom_single_repr = b + self.conditioned_transition_blocks[i](atom_single_repr, atom_single_proj)
        return atom_single_repr


class AtomAttentionEncoder(nn.Module):
    pass


class AtomAttentionDecoder(nn.Module):
    pass
