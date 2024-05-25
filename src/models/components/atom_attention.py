"""Sequence-local atom attention.
The 'sequence-local atom attention' represents the whole structure as a flat list of atoms and allows all atoms to
'talk' directly to each other within a certain sequence neighbourhood. e.g. each subset of 32 atoms attends to the
subset of the nearby 128 atoms (nearby in the sequence space). This gives the network the capacity to learn general
rules about local atom constellations, independently of the coarse-grained tokenization where each standard residue
is represented with a single token only."""
import torch
from torch import nn


class AttentionPairBias(nn.Module):
    """Implements the sequence-local atom attention with pair bias.
    This is implemented separately to the attention module that performs full self-attention
    since sequence-local atom attention requires a memory-efficient implementation.
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, bias=None):
        # Input projections
        pass


class AtomTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(self):
        pass


class AtomAttentionEncoder(nn.Module):
    pass


class AtomAttentionDecoder(nn.Module):
    pass
