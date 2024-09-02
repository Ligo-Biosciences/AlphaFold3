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

"""
Diffusion Module from AlphaFold3.
The StructureModule of AlphaFold2 using invariant point attention was replaced with a relatively standard
non-equivariant point-cloud diffusion model over all atoms. The denoiser is based on a modern transformer,
but with several modifications to make it more amenable to the task. The main changes are:
 - Conditioning from the trunk in several ways: we initialise the activations for the single embedding, use
    a variant of Adaptive Layernorm for the single conditioning and logit biasing for the pair conditioning.
 - Standard transformer tricks (e.g. SwiGLU) and methods used in AlphaFold2 (gating)
 - A two-level architecture, working first on atoms, then tokens, then atoms again.
"""
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm
from typing import Dict, Tuple
from src.models.diffusion_conditioning import DiffusionConditioning
from src.models.diffusion_transformer import DiffusionTransformer
from src.models.components.atom_attention import AtomAttentionEncoder, AtomAttentionDecoder
from src.models.components.primitives import LinearNoBias
from src.utils.geometry.vector import Vec3Array
from src.diffusion.augmentation import centre_random_augmentation
from src.diffusion.noise import sample_noise_level, noise_positions


class DiffusionModule(torch.nn.Module):
    def __init__(
            self,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_token: int = 768,
            c_tokenpair: int = 128,
            atom_encoder_blocks: int = 3,
            atom_encoder_heads: int = 16,
            dropout: float = 0.0,
            atom_attention_n_queries: int = 32,
            atom_attention_n_keys: int = 128,
            atom_decoder_blocks: int = 3,
            atom_decoder_heads: int = 16,
            token_transformer_blocks: int = 24,
            token_transformer_heads: int = 16,
            sd_data: float = 16.0,
            s_max: float = 160.0,
            s_min: float = 4e-4,
            p: float = 7.0,
            clear_cache_between_blocks: bool = False,
            blocks_per_ckpt: int = 1,
    ):
        super(DiffusionModule, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.atom_encoder_blocks = atom_encoder_blocks
        self.atom_encoder_heads = atom_encoder_heads
        self.dropout = dropout
        self.atom_attention_n_queries = atom_attention_n_queries
        self.atom_attention_n_keys = atom_attention_n_keys
        self.token_transformer_blocks = token_transformer_blocks
        self.token_transformer_heads = token_transformer_heads
        self.sd_data = sd_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.clear_cache_between_blocks = clear_cache_between_blocks
        self.blocks_per_ckpt = blocks_per_ckpt

        # Conditioning
        self.diffusion_conditioning = DiffusionConditioning(
            c_token=c_token,
            c_pair=c_tokenpair,
            sd_data=sd_data
        )

        # Sequence-local atom attention and aggregation to coarse-grained tokens
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_trunk_pair=c_tokenpair,
            no_blocks=atom_decoder_blocks,
            no_heads=atom_encoder_heads,
            dropout=dropout,
            n_queries=atom_attention_n_queries,
            n_keys=atom_attention_n_keys,
            trunk_conditioning=True,
            clear_cache_between_blocks=clear_cache_between_blocks
        )

        # Full self-attention on token level
        self.token_proj = nn.Sequential(
            LayerNorm(c_token),
            LinearNoBias(c_token, c_token, init='final')
        )
        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token,
            c_pair=c_tokenpair,
            no_blocks=token_transformer_blocks,
            no_heads=token_transformer_heads,
            dropout=dropout,
            clear_cache_between_blocks=clear_cache_between_blocks,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.token_post_layer_norm = LayerNorm(c_token)

        # Broadcast token activations to atoms and run sequence-local atom attention
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            no_blocks=atom_decoder_blocks,
            no_heads=atom_decoder_heads,
            dropout=dropout,
            n_queries=atom_attention_n_queries,
            n_keys=atom_attention_n_keys
        )
    
    def c_skip(self, timesteps: Tensor) -> Tensor:
        """Computes the skip connection scaling factor from Karras et al. (2022)."""
        return self.sd_data ** 2 / (self.sd_data ** 2 + timesteps ** 2)
    
    def c_out(self, timesteps: Tensor) -> Tensor:
        """Computes the output scaling factor from Karras et al. (2022)."""
        return timesteps * self.sd_data / torch.sqrt(self.sd_data ** 2 + timesteps ** 2)
    
    def c_in(self, timesteps: Tensor) -> Tensor:
        """Computes the input scaling factor from Karras et al. (2022)."""
        return 1. / torch.sqrt(self.sd_data ** 2 + timesteps ** 2)

    def scale_inputs(
            self,
            noisy_atoms: Tensor,
            timesteps: Tensor
    ) -> Tensor:
        """Scales positions to dimensionless vectors with approximately unit variance.
        Args:
            noisy_atoms:
                [bs, S, n_atoms, 3] tensor of noisy atom positions
            timesteps:
                [bs, S, 1] tensor of timesteps
        Returns:
            [bs, S, n_atoms, 3] rescaled noisy atom positions
        """
        c_in = self.c_in(timesteps).unsqueeze(-1)  # (bs, S, 1)
        return c_in * noisy_atoms 

    def rescale_with_updates(
            self,
            r_updates: Tensor,
            noisy_atoms: Tensor,
            timesteps: Tensor
    ) -> Tensor:
        """
        Rescales updates to positions and combines with input positions.
        Args:
            r_updates:
                [bs, S, n_atoms, 3] updates to the atom positions from the network
            noisy_atoms:
                [bs, S, n_atoms, 3] noisy atom positions
            timesteps:
                [bs, S, 1] timestep tensor
        Return:
            [bs, S, n_atoms, 3] updated atom positions
        """
        c_skip = self.c_skip(timesteps).unsqueeze(-1)  # (bs, S, 1, 1)
        c_out = self.c_out(timesteps).unsqueeze(-1)  # (bs, S, 1, 1)
        return c_skip * noisy_atoms + c_out * r_updates 

    def forward(
            self,
            noisy_atoms: Tensor,  # (bs, S, n_atoms, 3)
            timesteps: Tensor,  # (bs, S, 1)
            features: Dict[str, Tensor],  # input feature dict
            s_inputs: Tensor,  # (bs, n_tokens, c_token)
            s_trunk: Tensor,  # (bs, n_tokens, c_token)
            z_trunk: Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            use_deepspeed_evo_attention: bool = True
    ) -> Tensor:
        """Single denoising step that denoises atomic coordinates based on conditioning.
        Args:
            noisy_atoms:
                [bs, S, n_atoms, 3] tensor of noisy atom positions where S is the
                samples_per_trunk dimension.
            timesteps:
                tensor of timesteps (bs, S, 1) where S is the samples_per_trunk dimension.
            features:
                input feature dictionary containing the tensors:
                    "ref_pos" ([*, N_atoms, 3]):
                        atom positions in the reference conformers, with
                        a random rotation and translation applied. Atom positions in Angstroms.
                    "ref_charge" ([*, N_atoms]):
                        Charge for each atom in the reference conformer.
                    "ref_mask" ([*, N_atoms]):
                        Mask indicating which atom slots are used in the reference
                        conformer.
                    "ref_element" ([*, N_atoms, 128]):
                        One-hot encoding of the element atomic number for each atom
                        in the reference conformer, up to atomic number 128.
                    "ref_atom_name_chars" ([*, N_atom, 4, 64]):
                        One-hot encoding of the unique atom names in the reference
                        conformer. Each character is encoded as ord(c - 32), and names are padded to
                        length 4.
                    "ref_space_uid" ([*, N_atoms]):
                        Numerical encoding of the chain id and residue index associated
                        with this reference conformer. Each (chain id, residue index) tuple is assigned
                        an integer on first appearance.
                    "atom_to_token" ([*, N_atoms]):
                        Token index for each atom in the flat atom representation.
                    "residue_index" ([*, N_tokens]):
                        Residue number in the token's original input chain.
                    "token_index" ([*, N_tokens]):
                        Token number. Increases monotonically; does not restart at 1
                        for new chains.
                    "asym_id" ([*, N_tokens]):
                        Unique integer for each distinct chain.
                    "entity_id" ([*, N_tokens]):
                        Unique integer for each distinct sequence.
                    "sym_id" ([*, N_tokens]):
                        Unique integer within chains of this sequence. E.g. if chains
                        A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
                    "token_mask" ([*, N_tokens]):
                        [*, N_tokens] binary mask for tokens, whether token is present (not padding)
                    "atom_mask" ([*, N_atoms]):
                        binary mask for atoms, whether atom is present (will still be 1.0 if
                        atom is missing from the crystal structure, only 0.0 for padding)
            s_inputs:
                [*, n_tokens, c_token] Single conditioning input
            s_trunk:
                [*, n_tokens, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, n_tokens, n_tokens, c_pair] Pair conditioning from Pairformer trunk
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's optimized kernel for the attention

        """
        # Grab data about the inputs
        *_, n_tokens = features["asym_id"].shape

        # Extract masks
        token_mask = features["token_mask"]  # (bs, n_tokens)
        atom_mask = features["atom_mask"]  # (bs, n_atoms)

        # Conditioning
        token_repr, pair_repr = self.diffusion_conditioning(
            timesteps, features, s_inputs, s_trunk, z_trunk, token_mask
        )

        # Scale positions to dimensionless vectors with approximately unit variance
        r_noisy = self.scale_inputs(noisy_atoms, timesteps)

        # Sequence local atom attention and aggregation to coarse-grained tokens
        atom_encoder_output = self.atom_attention_encoder(
            features=features,
            n_tokens=n_tokens,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noisy_pos=r_noisy,
            mask=atom_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )

        # Full self-attention on token level
        token_single = atom_encoder_output.token_single + self.token_proj(token_repr)
        token_single = self.diffusion_transformer(
            single_repr=token_single,
            single_proj=token_repr,
            pair_repr=pair_repr,
            mask=token_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )

        token_single = self.token_post_layer_norm(token_single)

        # Broadcast token activations to atoms and run sequence-local atom attention
        atom_pos_updates = self.atom_attention_decoder(
            token_repr=token_single,  # (bs, S, n_tokens, c_token)
            atom_single_skip_repr=atom_encoder_output.atom_single_skip_repr,  # (bs, S, n_atoms, c_atom)
            atom_single_skip_proj=atom_encoder_output.atom_single_skip_proj,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr=atom_encoder_output.atom_pair_skip_repr,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx=features["atom_to_token"],  # (bs, n_atoms)
            mask=atom_mask,  # (bs, n_atoms)
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )  # (bs, S, n_atoms, 3)

        # Rescale updates to positions and combine with input positions
        output_pos = self.rescale_with_updates(atom_pos_updates, noisy_atoms, timesteps)
        return output_pos

    def train_step(
            self,
            ground_truth_atoms: Tensor,
            features: Dict[str, Tensor],
            s_inputs: Tensor,
            s_trunk: Tensor,
            z_trunk: Tensor,
            samples_per_trunk: int,
            use_deepspeed_evo_attention: bool = True
    ) -> Dict[str, Tensor]:
        """Train step of DiffusionModule.
        Args:
            ground_truth_atoms:
                The coordinates of the ground truth atoms. [*, N_atoms, 3]
            features:
                input feature dictionary containing the tensors:
                    "ref_pos" ([*, N_atoms, 3]):
                        atom positions in the reference conformers, with
                        a random rotation and translation applied. Atom positions in Angstroms.
                    "ref_charge" ([*, N_atoms]):
                        Charge for each atom in the reference conformer.
                    "ref_mask" ([*, N_atoms]):
                        Mask indicating which atom slots are used in the reference
                        conformer.
                    "ref_element" ([*, N_atoms, 128]):
                        One-hot encoding of the element atomic number for each atom
                        in the reference conformer, up to atomic number 128.
                    "ref_atom_name_chars" ([*, N_atom, 4, 64]):
                        One-hot encoding of the unique atom names in the reference
                        conformer. Each character is encoded as ord(c - 32), and names are padded to
                        length 4.
                    "ref_space_uid" ([*, N_atoms]):
                        Numerical encoding of the chain id and residue index associated
                        with this reference conformer. Each (chain id, residue index) tuple is assigned
                        an integer on first appearance.
                    "atom_to_token" ([*, N_atoms]):
                        Token index for each atom in the flat atom representation.
                    "residue_index" ([*, N_tokens]):
                        Residue number in the token's original input chain.
                    "token_index" ([*, N_tokens]):
                        Token number. Increases monotonically; does not restart at 1
                        for new chains.
                    "asym_id" ([*, N_tokens]):
                        Unique integer for each distinct chain.
                    "entity_id" ([*, N_tokens]):
                        Unique integer for each distinct sequence.
                    "sym_id" ([*, N_tokens]):
                        Unique integer within chains of this sequence. E.g. if chains
                        A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
                    "token_mask" ([*, N_tokens]):
                        [*, N_tokens] binary mask for tokens, whether token is present (not padding)
                    "atom_mask" ([*, N_atoms]):
                        binary mask for atoms, whether atom is present (will still be 1.0 if
                        atom is missing from the crystal structure, only 0.0 for padding)
            s_inputs:
                [*, N_token, c_token] Single conditioning input
            s_trunk:
                [*, N_token, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, N_token, N_token, c_pair] Pair conditioning from Pairformer trunk
            samples_per_trunk:
                the number of diffusion samples per trunk embedding.
                Total samples = batch_size * samples_per_trunk
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's Evoformer attention kernels
        """
        # Grab data about the inputs
        batch_size, n_atoms, _ = ground_truth_atoms.shape
        device, dtype = s_inputs.device, s_inputs.dtype
        atom_mask = features["atom_mask"][..., None, :].expand(batch_size, samples_per_trunk, n_atoms)

        # Create samples_per_trunk noisy versions of the ground truth atoms
        timesteps = sample_noise_level((batch_size, samples_per_trunk, 1), device=device, dtype=dtype)
        ground_truth_atoms = Vec3Array.from_array(  # expand to (bs, S, n_atoms, 3)
            ground_truth_atoms.unsqueeze(-3).expand(-1, -1, samples_per_trunk, n_atoms, 3)
        )

        # Randomly rotate each replica of the ground truth atoms
        aug_gt_atoms = centre_random_augmentation(ground_truth_atoms, atom_mask)

        # Noise the ground truth atoms
        noisy_atoms = noise_positions(aug_gt_atoms, timesteps)

        # Run the denoising step
        denoised_atoms = self.forward(
            noisy_atoms=noisy_atoms.to_tensor().to(dtype),
            timesteps=timesteps,
            features=features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention
        )
        outputs = {
            "denoised_atoms": denoised_atoms,
            "timesteps": timesteps,
            "augmented_gt_atoms": aug_gt_atoms.to_tensor().to(dtype)
        }
        return outputs

    @torch.no_grad()
    def sample(
            self,
            features: Dict[str, Tensor],  # input feature dict
            s_inputs: Tensor,  # (bs, n_tokens, c_token)
            s_trunk: Tensor,  # (bs, n_tokens, c_token)
            z_trunk: Tensor,  # (bs, n_tokens, n_tokens, c_token)
            n_steps: int = 200,
            samples_per_trunk: int = 32,
            gamma_0: float = 0.8,
            gamma_min: float = 1.0,
            noise_scale: float = 1.003,
            step_scale: float = 1.5,
            use_deepspeed_evo_attention: bool = False
    ) -> Tensor:
        """Implements SampleDiffusion, Algorithm 18 in AlphaFold3 Supplement.
        Args:
            features:
                input feature dictionary containing the tensors:
                    "ref_pos" ([*, N_atoms, 3]):
                        atom positions in the reference conformers, with
                        a random rotation and translation applied. Atom positions in Angstroms.
                    "ref_charge" ([*, N_atoms]):
                        Charge for each atom in the reference conformer.
                    "ref_mask" ([*, N_atoms]):
                        Mask indicating which atom slots are used in the reference
                        conformer.
                    "ref_element" ([*, N_atoms, 128]):
                        One-hot encoding of the element atomic number for each atom
                        in the reference conformer, up to atomic number 128.
                    "ref_atom_name_chars" ([*, N_atom, 4, 64]):
                        One-hot encoding of the unique atom names in the reference
                        conformer. Each character is encoded as ord(c - 32), and names are padded to
                        length 4.
                    "ref_space_uid" ([*, N_atoms]):
                        Numerical encoding of the chain id and residue index associated
                        with this reference conformer. Each (chain id, residue index) tuple is assigned
                        an integer on first appearance.
                    "atom_to_token" ([*, N_atoms]):
                        Token index for each atom in the flat atom representation.
                    "residue_index" ([*, N_tokens]):
                        Residue number in the token's original input chain.
                    "token_index" ([*, N_tokens]):
                        Token number. Increases monotonically; does not restart at 1
                        for new chains.
                    "asym_id" ([*, N_tokens]):
                        Unique integer for each distinct chain.
                    "entity_id" ([*, N_tokens]):
                        Unique integer for each distinct sequence.
                    "sym_id" ([*, N_tokens]):
                        Unique integer within chains of this sequence. E.g. if chains
                        A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]
                    "token_mask" ([*, N_tokens]):
                        [*, N_tokens] binary mask for tokens, whether token is present (not padding)
                    "atom_mask" ([*, N_atoms]):
                        binary mask for atoms, whether atom is present (will still be 1.0 if
                        atom is missing from the crystal structure, only 0.0 for padding)
            s_inputs:
                [*, N_token, c_token] Single conditioning input
            s_trunk:
                [*, N_token, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, N_token, N_token, c_pair] Pair conditioning from Pairformer trunk
            samples_per_trunk:
                the number of diffusion samples per trunk embedding.
                Total samples = batch_size * samples_per_trunk
            n_steps:
                number of diffusion steps
            gamma_0:
                lower bound on the gamma. Defaults to value used in the paper.
            gamma_min:
                threshold for the lower bound on gamma. Defaults to value used in the paper.
            noise_scale:
                noise scale. Defaults to value used in the paper.
            step_scale:
                step scale. Defaults to value used in the paper.
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's Evoformer attention kernel.
        Returns:
            [bs, samples_per_trunk, n_atoms, 3] sampled coordinates
        """

        # Grab data about the input
        batch_size, n_atoms, _ = features["ref_pos"].shape
        dtype, device = s_inputs.dtype, s_inputs.device
        atom_mask = features["atom_mask"][..., None, :].expand(batch_size, samples_per_trunk, n_atoms)

        # Create the noise schedule with float64 dtype to prevent numerical issues
        t = torch.linspace(0, 1, n_steps, device=device, dtype=torch.float64).unsqueeze(-1)  # (n_steps, 1)
        s_max_root = math.pow(self.s_max, 1 / self.p)
        s_min_root = math.pow(self.s_min, 1 / self.p)
        noise_schedule = self.sd_data * (s_max_root + t * (s_min_root - s_max_root)) ** self.p

        # Sample random noise as the initial structure
        x_l = noise_schedule[0] * Vec3Array.randn((batch_size, samples_per_trunk, n_atoms), device)  # float32

        for i in range(1, n_steps):
            # Centre random augmentation
            x_l = centre_random_augmentation(x_l, atom_mask)

            c_step = noise_schedule[i]
            prev_step = noise_schedule[i - 1]
            gamma = gamma_0 if c_step > gamma_min else 0.0

            # Expand c_step and prev_step for proper broadcasting
            c_step = c_step[None, None, ...].expand(batch_size, samples_per_trunk, 1)  # (bs, samples_per_trunk, 1)
            prev_step = prev_step[None, None, ...].expand(batch_size, samples_per_trunk, 1)

            t_hat = torch.mul(prev_step, torch.add(gamma, 1.0))
            normal_noise = Vec3Array.randn((batch_size, samples_per_trunk, n_atoms), device)
            zeta_factor = (noise_scale * torch.sqrt((t_hat ** 2 - prev_step ** 2))).to(normal_noise.x.dtype)
            zeta = zeta_factor * normal_noise
            x_noisy = x_l + zeta

            # Run DiffusionModule to denoise structure
            x_denoised = self.forward(
                noisy_atoms=x_noisy.to_tensor().to(dtype),  # revert to model dtype
                timesteps=t_hat.to(dtype),  # (bs, samples_per_trunk, 1)
                features=features,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention
            )
            # Back to Vec3Array (float32)
            x_denoised = Vec3Array.from_array(x_denoised)

            # Update the noisy structure
            delta = (x_l - x_denoised) / t_hat
            dt = c_step - t_hat
            x_l = x_noisy + step_scale * dt * delta

        return x_l.to_tensor().to(dtype)  # revert to model dtype
