"""Sampling from the diffusion trajectory defined by AlphaFold3 during training and inference."""

import torch
from src.utils.geometry.vector import Vec3Array


# Training
def noise_positions(
        atom_positions: Vec3Array,  # (bs, n_atoms)
        noise_levels: torch.Tensor  # (bs, 1)
) -> Vec3Array:  # (bs, n_atoms)
    """Sample from the diffusion trajectory with Gaussian noise."""
    batch_size, n_atoms = atom_positions.shape
    device = atom_positions.x.device

    # X = (y + n) where y is clean signal and n ~ N(0, noise_level^2)
    noised_pos = atom_positions + noise_levels * Vec3Array.randn((batch_size, n_atoms), device)
    return noised_pos


def sample_noise_level(
        random_normal: torch.Tensor,
        sd_data: float = 16.0
) -> torch.Tensor:
    """Sample noise level given random normal noise.
    The sampled noise level has the same shape and device as the input."""
    return torch.mul(torch.exp(torch.add(torch.mul(random_normal, 1.5), -1.2)), sd_data)

# Inference
