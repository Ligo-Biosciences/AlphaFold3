"""Sampling from the diffusion trajectory defined by AlphaFold3 during training and inference."""
from pytorch_lightning import LightningModule
import torch
from src.utils.geometry.vector import Vec3Array
from typing import Tuple, Dict
from src.diffusion.augmentation import centre_random_augmentation


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
def sample_diffusion(
        model: LightningModule,  # denoising model
        features: Dict[str, torch.Tensor],  # input feature dict
        # s_inputs: torch.Tensor,  # (bs, n_tokens, c_token)
        # s_trunk: torch.Tensor,  # (bs, n_tokens, c_token)
        # z_trunk: torch.Tensor,  # (bs, n_tokens, n_tokens, c_token)çö
        noise_schedule: torch.Tensor,  # (n_steps, 1)
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5
) -> Vec3Array:
    """Sample from the diffusion trajectory with Gaussian noise.
    This method follows the notation in Algorithm 18 of AlphaFold3
    Supp. Info.
    """
    batch_size, n_atoms, _ = features["ref_pos"].shape
    device = model.device

    # Sample random noise as the initial structure
    x_l = Vec3Array.randn((batch_size, n_atoms), device)

    for i in range(1, noise_schedule.shape[0]):
        # Centre random augmentation
        x_l = centre_random_augmentation(x_l)
        c_step = noise_schedule[i].unsqueeze(0).expand(batch_size, 1)  # shape == (bs, 1)
        prev_step = noise_schedule[i - 1].unsqueeze(0).expand(batch_size, 1)  # shape == (bs, 1)

        gamma = gamma_0 if c_step > gamma_min else 0.0
        t_hat = torch.mul(prev_step, torch.add(gamma, 1.0))
        zeta = noise_scale * torch.sqrt((t_hat ** 2 - prev_step ** 2)) * Vec3Array.randn((batch_size, n_atoms), device)
        x_noisy = x_l + zeta

        # Compute the denoised structure
        x_denoised = model.forward(noisy_atoms=x_noisy,  # (bs, n_atoms)
                                   timesteps=t_hat,  # (bs, 1)
                                   features=features)

        # Update the noisy structure
        delta = (x_l - x_denoised) / t_hat
        dt = c_step - t_hat
        x_l = x_noisy + step_scale * dt * delta

    # Return the denoised structure
    return x_l
