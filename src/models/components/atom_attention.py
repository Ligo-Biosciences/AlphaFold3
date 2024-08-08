import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Dict, NamedTuple, Optional, Tuple
from src.models.components.primitives import AdaLN, Linear, LinearNoBias, Attention, LayerNorm
from src.models.components.transition import ConditionedTransitionBlock
from einops import rearrange
from functools import partial
from src.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn

checkpoint = get_checkpoint_fn()


def partition_tensor(
        x: Tensor,  # (*, n_atoms, c)
        n_queries: int = 32,
        n_keys: int = 128,
        pad_value: Optional[float] = None,
) -> Tensor:
    """Partitions the input flat atom tensor into windows of n_keys with a slide stride of n_queries.
    The input tensor is padded to make the centers of the partitions align with the subset centers in AlphaFold3.
    Subset centers = (15.5, 47.5, 79.5, ...)
    """
    # Pad
    pad = n_keys // 2 - n_queries // 2
    x = pad_column(x, (pad, pad), mode='constant', value=pad_value)

    # Sliding window along n_atoms dimension
    windows = x.unfold(dimension=-2, size=n_keys, step=n_queries)
    return windows.transpose(-1, -2)  # unfold reverses channel dimension, undo this


def pad_column(x, pad, mode='constant', value=None) -> Tensor:
    """Applies padding to the second to last dimension of a tensor."""
    return F.pad(x.transpose(-1, -2), pad, mode=mode, value=value).transpose(-1, -2)


def extract_locals(
        bias_tensor: Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        pad_value: Optional[float] = -1e4
) -> Tensor:
    """Extracts biases etc. that are local in the sequence space. Also pads the local biases with large negative values
    to mask the gaps during attention computation.

        Args:
            bias_tensor:
                A tensor of shape [batch_size, N_atoms, N_atoms, channels].
            n_queries:
                The increment between the centers of the partitions.
            n_keys:
                The length of the partitions.
            pad_value:
                The value to use for padding.
        Returns:
            A tensor of shape [batch_size, N_atoms // partition_increment, partition_length, channels].
    """
    batch_size, n_atoms, _, channels = bias_tensor.shape
    # Pad bias tensor column-wise by n_keys // 2 - n_queries // 2 on each side
    pad = n_keys // 2 - n_queries // 2
    bias_tensor = pad_column(bias_tensor, (pad, pad), mode='constant', value=pad_value)

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
    """
    Implements the sequence-local atom attention with pair bias.
    This is implemented separately to the attention module that performs full self-attention
    since sequence-local atom attention requires a memory-efficient implementation.
    """

    def __init__(
            self,
            c_atom: int,
            c_atompair: int,
            no_heads: int,
            dropout: float = 0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            inf: float = 1e8
    ):
        """
        Initialize the AtomAttentionPairBias module.
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
        super(AtomAttentionPairBias, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.no_heads = no_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.inf = inf

        # Projections
        self.ada_ln = AdaLN(c_atom)
        self.output_proj_linear = Linear(c_atom, c_atom, init='gating')
        self.output_proj_linear.bias = nn.Parameter(torch.ones(c_atom) * -2.0)  # gate values will be ~0.11

        # Attention
        self.attention = Attention(
            c_q=c_atom,
            c_k=c_atom,
            c_v=c_atom,
            c_hidden=c_atom // no_heads,
            no_heads=no_heads,
            gating=True
        )

        # Pair bias
        self.layer_norm_pair = LayerNorm(self.c_atompair)
        self.linear_pair = LinearNoBias(self.c_atompair, self.no_heads, init='default')

    def _prep_biases(
            self,
            atom_single: Tensor,  # (bs, S, n_atoms, c_atom)
            atom_pair_local: Tensor,  # (bs, n_atoms // n_queries, n_queries, n_keys, c_atompair)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
    ):
        """
        Prepares the mask and pair biases in the shapes expected by the DS4Science attention.
        Args:
            atom_single:
                [bs, S, n_atoms, c_atom] atom single representation where S is the samples per trunk dimension.
            atom_pair_local:
                [bs, n_atoms // n_queries, n_queries, n_keys, c_atompair] local atom pair representation tensor.
                The pair representation is partitioned into n_atoms // n_queries for memory efficiency instead of
                the full N_atoms x N_atoms
            mask:
                [bs, n_atoms] atom mask tensor where 1.0 indicates atom to be attended and
                0.0 indicates atom not to be attended. The mask is shared across the S dimension.

        Expected shapes for the DS4Science kernel:
        # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
        # res_mask: [Batch, N_seq, 1, 1, N_res]
        # pair_bias: [Batch, 1, Head, N_res, N_res]

        # TODO: with the current implementation of DS4Science kernel, we can only use it if n_queries == n_keys.
        """
        # Compute the single mask
        n_seq, n_atoms, _ = atom_single.shape[-3:]
        if mask is None:
            # [*, N_seq, N_atoms]
            mask = atom_single.new_ones(
                atom_single.shape[:-3] + (n_seq, n_atoms),
            )
        else:
            # Expand mask by N_seq (or samples per trunk)
            new_shape = (mask.shape[:-1] + (n_seq, n_atoms))  # (*, N_seq, N_atoms)
            mask = mask.unsqueeze(-2).expand(new_shape)  # (bs, N_seq, N_atoms)
            mask = mask.to(atom_single.dtype)

        # Partition mask,  Target mask shape: [bs, n_atoms // n_queries, S, n_keys]
        mask = partition_tensor(mask[..., None], self.n_queries, self.n_keys)  # (bs, S, n_atoms // 32, 128, 1)
        mask = mask.squeeze(-1)  # (bs, S, n_atoms // 32, 128)
        mask = rearrange(mask, pattern='b s p k -> (b p) s k')

        # [*, N_seq, 1, 1, N_keys]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # Project pair biases from head representation
        local_pair_b = self.linear_pair(self.layer_norm_pair(atom_pair_local))

        # [*, 1, Head, N_queries, N_keys]
        pair_bias = rearrange(local_pair_b, 'b p q k h -> (b p) h q k')
        pair_bias = pair_bias.unsqueeze(-4)
        return mask_bias, pair_bias

    def forward(
            self,
            atom_single: Tensor,  # (bs, S, n_atoms, c_atom)
            atom_proj: Tensor,  # (bs, S, n_atoms, c_atom)
            atom_pair_local: Tensor,  # (bs, n_atoms // n_queries, n_queries, n_keys, c_atompair)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
            use_deepspeed_evo_attention: bool = False,
    ):
        """
        Attention mechanism for sequence-local atom attention.
        Args:
            atom_single:
                [bs, S, n_atoms, c_atom] atom single representation where S is the samples per trunk dimension.
            atom_proj:
                [bs, S, n_atoms, c_atom] atom projection representation where S is the samples per trunk dimension.
            atom_pair_local:
                [bs, n_atoms // n_queries, n_queries, n_keys, c_atompair] local atom pair representation tensor.
                The pair representation is partitioned into n_atoms // n_queries for memory efficiency instead of
                the full N_atoms x N_atoms
            mask:
                [bs, n_atoms] atom mask tensor where 1.0 indicates atom to be attended and
                0.0 indicates atom not to be attended. The mask is shared across the S dimension.
            use_deepspeed_evo_attention:
                whether to use Deepspeed's optimized kernel for the attention. It is only usable
                here if n_queries == n_keys.

        Returns:
            [bs, S, n_atoms, c_atom] updated atom single representation
        """
        if use_deepspeed_evo_attention:
            assert self.n_queries == self.n_keys, \
                "DeepSpeed Evo Attention is only valid for n_queries == n_keys within atom attention."
        bs, S, n_atoms, _ = atom_single.shape

        # Input projection
        a = self.ada_ln(atom_single, atom_proj)  # (bs, S, n_atoms, c_atom)

        # Prep biases
        mask_bias, pair_bias = self._prep_biases(atom_single, atom_pair_local, mask)

        # Partition and reshape
        atom_qx = partition_tensor(a, self.n_queries, self.n_queries)  # (bs, S, n_atoms // 32, 32, c_atom)
        atom_kvx = partition_tensor(a, self.n_queries, self.n_keys)  # (bs, S, n_atoms // 32, 128, c_atom)
        atom_qx = rearrange(atom_qx, 'b s p q c -> (b p) s q c')
        atom_kvx = rearrange(atom_kvx, 'b s p k c -> (b p) s k c')

        # Attention
        output = self.attention(
            q_x=atom_qx,
            kv_x=atom_kvx,
            biases=[mask_bias, pair_bias],
        )  # (bs * n_atoms // n_queries, S, n_queries, c_atom)

        # Reshape back to original, (bs, n_atoms // n_queries, S, n_queries, c_atom)
        output = output.reshape(bs, -1, S, self.n_queries, self.c_atom)
        output = rearrange(output, 'b p s q c -> b s (p q) c')  # (bs, S, n_atoms, c_atom)

        # Output projection
        output = F.sigmoid(self.output_proj_linear(output)) * output
        return output


class AtomTransformerBlock(nn.Module):
    def __init__(
            self,
            c_atom: int,
            c_atompair: int = 16,
            no_heads: int = 8,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
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
        self.atom_attention = AtomAttentionPairBias(
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_heads=no_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
        )
        self.transition = ConditionedTransitionBlock(c_atom)

    def forward(
            self,
            atom_single: Tensor,
            atom_proj: Tensor,
            atom_pair_local: Tensor,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        a = self.atom_attention(atom_single, atom_proj, atom_pair_local, mask)
        atom_single = a + self.transition(atom_single, atom_proj)
        return atom_single, atom_proj, atom_pair_local


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
            compile_module: bool = False,
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
            compile_module:
                Whether to compile the module.
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
        self.compile_module = compile_module

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
            atom_pair_local: Tensor,
            mask: Optional[Tensor] = None,
    ):
        """Prepare the input tensors for each AtomTransformerBlock."""
        blocks = [
            partial(
                block if not self.compile_module else torch.compile(block),
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
            atom_single: Tensor,
            atom_proj: Tensor,
            atom_pair_local: Tensor,
            mask: Optional[Tensor] = None,
    ):
        """
        Forward pass of the AtomTransformer module. Algorithm 23 in AlphaFold3 supplement.
        Args:
            atom_single:
                [bs, S, n_atoms, c_atom] atom single representation where S is the samples per trunk dimension.
            atom_proj:
                [bs, n_atoms, c_atom] atom projection representation.
            atom_pair_local:
                [bs, n_atoms // n_queries, n_queries, n_keys, c_atompair] local atom pair representation tensor.
                The pair representation is partitioned into n_atoms // n_queries for memory efficiency instead of
                the full N_atoms x N_atoms
            mask:
                [bs, n_atoms] atom mask tensor where 1.0 indicates atom to be attended and
                0.0 indicates atom not to be attended. The mask is shared across the S dimension.
        """
        # Expand atom_proj for proper broadcasting
        atom_proj = atom_proj.unsqueeze(-3)

        blocks = self._prep_blocks(
            atom_single=atom_single,
            atom_proj=atom_proj,
            atom_pair_local=atom_pair_local,
            mask=mask,
        )
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        atom_single, atom_proj, atom_pair_local = checkpoint_blocks(
            blocks,
            args=(atom_single, atom_proj, atom_pair_local),
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


def map_token_pairs_to_local_atom_pairs(
        token_pairs: Tensor,
        tok_idx: Tensor,
        n_queries=32,
        n_keys=128
) -> Tensor:
    """Given token pairs and token indices, map token pairs to local atom pairs to be used within local atom attention.
    Args:
        token_pairs:
            [bs, n_tokens, n_tokens, c_pair] pair representation from the trunk.
        tok_idx:
            [bs, n_atoms] Tensor containing token indices per atom.
        n_queries:
            The size of the atom window. Defaults to 32.
        n_keys:
            Number of atoms each atom attends to in local sequence space. Defaults to 128.

    Returns:
        [bs, n_atoms // n_queries, n_queries, n_keys, c_pair] tensor containing atom pair embeddings derived from token
        pair embeddings. For each atom pair (l, m), the corresponding token pair's embeddings are extracted."""
    bs, n_atoms = tok_idx.shape
    _, n_tokens, _, c_pair = token_pairs.shape
    tok_idx = tok_idx.long()  # convert to int for indexing

    # Expand tok_idx for efficient gather operation
    tok_idx_l = tok_idx.unsqueeze(2).expand(-1, -1, n_atoms).unsqueeze(-1)
    tok_idx_m = tok_idx.unsqueeze(1).expand(-1, n_atoms, -1).unsqueeze(-1)
    batch_index = torch.arange(bs).reshape(bs, 1, 1, 1)

    # Extract the local indices
    local_tok_idx_l = extract_locals(tok_idx_l, n_queries=n_queries, n_keys=n_keys, pad_value=0).squeeze(-1)
    local_tok_idx_m = extract_locals(tok_idx_m, n_queries=n_queries, n_keys=n_keys, pad_value=0).squeeze(-1)

    # Gather token pair embeddings using advanced indexing
    local_atom_pairs = token_pairs[batch_index, local_tok_idx_l, local_tok_idx_m, :]

    return local_atom_pairs


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


class AtomAttentionEncoderOutput(NamedTuple):
    """Structured output class for AtomAttentionEncoder."""
    token_single: torch.Tensor  # (bs, n_tokens, c_token)
    atom_single_skip_repr: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_single_skip_proj: torch.Tensor  # (bs, n_atoms, c_atom)
    atom_pair_skip_repr: torch.Tensor  # (bs, n_atoms // n_queries, n_queries, n_keys, c_atompair)  TODO: local biases


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
            clear_cache_between_blocks: bool = False,
            compile_module: bool = False,
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
                compile_module:
                    Whether to compile the module.
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
        self.compile_module = compile_module

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
                LinearNoBias(c_token, c_atom, init='final')
            )
            self.proj_trunk_pair = nn.Sequential(
                LayerNorm(c_trunk_pair),
                LinearNoBias(c_trunk_pair, c_atompair, init='final')
            )

            self.linear_noisy_pos = LinearNoBias(3, c_atom, init='final')

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
            clear_cache_between_blocks=clear_cache_between_blocks,
            compile_module=compile_module
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
            [bs, n_atoms // n_queries, n_queries, n_keys, c_atompair] The pair representation
        """
        # Compute offsets between atom reference positions
        a = partition_tensor(features['ref_pos'], self.n_queries, self.n_queries)  # (bs, n_atoms // 32, 32, 3)
        b = partition_tensor(features['ref_pos'], self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128, 3)
        offsets = a[:, :, :, None, :] - b[:, :, None, :, :]  # (bs, n_atoms // 32, 32, 128, 3)

        # Compute the valid mask
        ref_space_uid = features['ref_space_uid'].unsqueeze(-1)  # (bs, n_atoms, 1)
        a = partition_tensor(ref_space_uid, self.n_queries, self.n_queries)  # (bs, n_atoms // 32, 32)
        b = partition_tensor(ref_space_uid, self.n_queries, self.n_keys)  # (bs, n_atoms // 32, 128)
        valid_mask = a[:, :, :, None] == b[:, :, None, :]  # (bs, n_atoms // 32, 32, 128, 1)
        valid_mask = valid_mask.to(offsets.dtype)  # convert boolean to binary

        # Embed the atom offsets and the valid mask
        local_atom_pair = self.linear_atom_offsets(offsets) * valid_mask

        # Embed pairwise inverse squared distances, and the valid mask
        squared_distances = offsets.pow(2).sum(dim=-1, keepdim=True)  # (bs, n_atoms // 32, 32, 128, 1)
        inverse_dists = torch.reciprocal(torch.add(squared_distances, 1))
        local_atom_pair = local_atom_pair + self.linear_atom_distances(inverse_dists) * valid_mask
        local_atom_pair = local_atom_pair + self.linear_mask(valid_mask) * valid_mask

        # If provided, add trunk embeddings
        if self.trunk_conditioning:
            local_atom_pair = local_atom_pair + map_token_pairs_to_local_atom_pairs(
                self.proj_trunk_pair(z_trunk),
                features['atom_to_token']
            )

        # Add the combined single conditioning to the pair representation
        a = partition_tensor(self.linear_single_to_pair_row(F.relu(atom_cond)), self.n_queries, self.n_queries)
        b = partition_tensor(self.linear_single_to_pair_col(F.relu(atom_cond)), self.n_queries, self.n_keys)
        local_atom_pair = local_atom_pair + (a[:, :, :, None, :] + b[:, :, None, :, :])

        # Run a small MLP on the pair activations
        local_atom_pair = self.pair_mlp(local_atom_pair)
        return local_atom_pair

    def init_single_repr(
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
                [*, S, n_atoms, 3] the noisy atom positions where S is the
                samples_per_trunk dimension.
        Returns:
            atom_single:
                atom single representation of shape [*, S, n_atoms, c_atom]. If trunk conditioning is False,
                S == 1.
            atom_single_conditioning:
                atom conditioning representation of shape [*, n_atoms, c_atom]
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
                dim=2
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
                    [*, N_atoms // n_queries, n_queries, n_keys, c_atompair] atom pair representation
                    (denoted p_lm in AF3 Supplement)
        """
        # Initialize representations
        atom_single, atom_single_conditioning = checkpoint(self.init_single_repr, features, s_trunk, noisy_pos)
        local_atom_pair = checkpoint(self.init_pair_repr, features, atom_single_conditioning, z_trunk)

        # Cross attention transformer
        atom_single = self.atom_transformer(atom_single, atom_single_conditioning, local_atom_pair, mask)

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
            atom_pair_skip_repr=local_atom_pair,  # (bs, n_atoms // n_queries, n_queries, n_keys, c_atompair)
        )
        return output


class AtomAttentionDecoder(nn.Module):
    """AtomAttentionDecoder that broadcasts per-token activations to per-atom activations."""

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
            clear_cache_between_blocks: bool = False,
            compile_module: bool = False,
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
            compile_module:
                Whether to compile the module.

        """
        super().__init__()
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.num_blocks = no_blocks
        self.num_heads = no_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.compile_module = compile_module

        # Broadcast token to atom
        self.linear_atom = LinearNoBias(c_token, c_atom, init='default')

        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_blocks=no_blocks,
            no_heads=no_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            clear_cache_between_blocks=clear_cache_between_blocks,
            compile_module=compile_module
        )

        # Output
        self.linear_update = LinearNoBias(c_atom, 3, init='final')
        self.layer_norm = LayerNorm(c_atom)

    def forward(
            self,
            token_repr: Tensor,  # (bs, S, n_tokens, c_token)
            atom_single_skip_repr: Tensor,  # (bs, S, n_atoms, c_atom)
            atom_single_skip_proj: Tensor,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr: Tensor,  # (bs, n_atoms // n_queries, n_queries, n_keys, c_atom)
            tok_idx: Tensor,  # (bs, n_atoms)
            mask: Optional[Tensor] = None,  # (bs, n_atoms)
    ) -> Tensor:
        """AtomAttentionDecoder. Algorithm 6 in AlphaFold3 supplement.
        Args:
            token_repr:
                [bs, S, n_tokens, c_atom] Per-token activations.
                S is the samples_per_trunk dimension.
            atom_single_skip_repr:
                [bs, S, n_atoms, c_atom] Per-atom activations added as the skip connection.
                S is the samples_per_trunk dimension.
            atom_single_skip_proj:
                [bs, 1, n_atoms, c_atom] Per-atom activations provided to AtomTransformer.
            atom_pair_skip_repr:
                [bs, n_atoms // n_queries, n_queries, n_keys, c_atompair] Pair activations provided
                to AtomTransformer.
            tok_idx:
                [bs, n_atoms] Token indices that encode which token each atom belongs to.
            mask:
                [bs, n_atoms] Mask for the atom transformer.
        Returns:
            [bs, S, n_atoms, 3] a tensor of per-atom coordinate updates.
        """
        # Broadcast per-token activations to per-atom activations and add the skip connection
        bs, S, n_tokens, c_atom = token_repr.shape
        atom_single = self.linear_atom(  # vectorize to pretend S is the batch dimension for the gather op
            torch.vmap(gather_token_repr)(token_repr, tok_idx.unsqueeze(-2).expand(-1, S, -1))
        )
        atom_single = atom_single + atom_single_skip_repr  # (bs, S, n_atoms, c_atom)

        # Cross-attention transformer
        atom_single = self.atom_transformer(atom_single, atom_single_skip_proj, atom_pair_skip_repr, mask)

        # Map to positions update
        r_atom_update = self.linear_update(self.layer_norm(atom_single))
        return r_atom_update
