"""Diffusion conditioning."""

import torch
from torch import nn
import math


class FourierEmbedding(nn.Module):
    """Fourier embedding for diffusion conditioning."""
    def __init__(self, embed_dim):
        super(FourierEmbedding, self).__init__()
        self.embed_dim = embed_dim
        # Randomly generate weight/bias once before training
        self.weight = nn.Parameter(torch.randn((embed_dim,)))
        self.bias = nn.Parameter(torch.randn((embed_dim,)))

    def forward(self, t):
        """Compute embeddings"""
        return torch.cos(torch.tensor(2 * math.pi) * (t * self.weight + self.bias))

