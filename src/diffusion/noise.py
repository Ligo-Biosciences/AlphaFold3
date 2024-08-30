"""Sampling from the diffusion trajectory defined by AlphaFold3 during training and inference."""
import torch
from src.utils.geometry.vector import Vec3Array
from typing import Sequence


# Training
def noise_positions(
        atom_positions: Vec3Array,  # (*, n_atoms)
        noise_levels: torch.Tensor  # (*, 1)
) -> Vec3Array:  # (*, n_atoms)
    """Sample from the diffusion trajectory with Gaussian noise."""
    device = atom_positions.x.device
    # X = (y + n) where y is clean signal and n ~ N(0, noise_level^2)
    noise = Vec3Array.randn(atom_positions.shape, device)
    noised_pos = atom_positions + noise_levels * noise
    return noised_pos


def sample_noise_level(
        shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
        sd_data: float = 16.0,
) -> torch.Tensor:
    """Sample noise level given random normal noise."""
    random_normal = torch.randn(shape, device=device, dtype=dtype)
    # Calculate noise level using a log-normal distribution
    exponent = (random_normal * 1.5) - 1.2
    noise_level = torch.exp(exponent) * sd_data
    return noise_level
