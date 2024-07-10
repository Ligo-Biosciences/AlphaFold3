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
import torch
from torch import nn
from torch import Tensor
from typing import Dict, Optional
from src.models.diffusion_conditioning import DiffusionConditioning
from src.models.diffusion_transformer import DiffusionTransformer
from src.models.components.atom_attention import AtomAttentionEncoder, AtomAttentionDecoder
from src.models.components.primitives import LinearNoBias, LayerNorm
from src.utils.tensor_utils import tensor_tree_map
from src.utils.geometry.vector import Vec3Array
from src.diffusion.augmentation import centre_random_augmentation


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
            clear_cache_between_blocks: bool = False,
            compile_model: bool = True,
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
        self.clear_cache_between_blocks = clear_cache_between_blocks

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
            num_blocks=token_transformer_blocks,
            num_heads=token_transformer_heads,
            dropout=dropout,
            clear_cache_between_blocks=clear_cache_between_blocks
        )
        self.token_post_layer_norm = LayerNorm(c_token)

        # Broadcast token activations to atoms and run sequence-local atom attention
        self.atom_attention_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            num_blocks=atom_decoder_blocks,
            num_heads=atom_decoder_heads,
            dropout=dropout,
            n_queries=atom_attention_n_queries,
            n_keys=atom_attention_n_keys,
        )
        if compile_model:
            self.diffusion_conditioning = torch.compile(self.diffusion_conditioning)
            self.atom_attention_encoder = torch.compile(self.atom_attention_encoder)
            # diffusion_transformer = torch.compile(self.diffusion_transformer)
            self.atom_attention_decoder = torch.compile(self.atom_attention_decoder)

    def scale_inputs(
            self,
            noisy_atoms: Tensor,
            timesteps: Tensor
    ) -> Tensor:
        """Scales positions to dimensionless vectors with approximately unit variance.
        Args:
            noisy_atoms:
                [bs, n_atoms, 3] tensor of noisy atom positions
            timesteps:
                [bs, 1] tensor of timesteps
        Returns:
            [bs, n_atoms, 3] rescaled noisy atom positions
        """
        denominator = torch.sqrt(torch.add(timesteps ** 2, self.sd_data ** 2))  # (bs, 1)
        rescaled_noisy = noisy_atoms / denominator.unsqueeze(-1)  # (bs, n_atoms, 3)
        return rescaled_noisy

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
                [bs, n_atoms, 3] updates to the atom positions from the network
            noisy_atoms:
                [bs, n_atoms, 3] noisy atom positions
            timesteps:
                [bs, 1] timestep tensor
        Return:
            [bs, n_atoms, 3] updated atom positions
        """
        noisy_pos_scale = torch.div(
            self.sd_data ** 2,
            torch.add(timesteps ** 2, self.sd_data ** 2)
        )

        noisy_pos_scale = noisy_pos_scale.unsqueeze(-1)  # (bs, 1, 1)
        r_update_scale = torch.sqrt(noisy_pos_scale) * timesteps.unsqueeze(-1)
        return noisy_atoms * noisy_pos_scale + r_updates * r_update_scale

    def forward(
            self,
            noisy_atoms: Tensor,  # (bs, n_atoms, 3)
            timesteps: Tensor,  # (bs, 1)
            features: Dict[str, Tensor],  # input feature dict
            s_inputs: Tensor,  # (bs, n_tokens, c_token)
            s_trunk: Tensor,  # (bs, n_tokens, c_token)
            z_trunk: Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            use_deepspeed_evo_attention: bool = True
    ) -> Tensor:
        """Diffusion module that denoises atomic coordinates based on conditioning.
        Args:
            noisy_atoms:
                tensor of noisy atom positions (bs, n_atoms, 3)
            timesteps:
                tensor of timesteps (bs, 1)
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
                        Residue number in the token’s original x chain.
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
                [*, n_tokens, c_token] Single conditioning x
            s_trunk:
                [*, n_tokens, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, n_tokens, n_tokens, c_pair] Pair conditioning from Pairformer trunk
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's optimized kernel for attention pair bias

        """
        # Grab data about the inputs
        *_, n_tokens = features["asym_id"].shape

        # Extract masks
        token_mask = features["token_mask"]  # (bs, n_tokens)
        atom_mask = features["atom_mask"]  # (bs, n_atoms)

        # Conditioning
        token_repr, pair_repr = self.diffusion_conditioning(
            timesteps=timesteps,
            features=features,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            mask=token_mask
        )

        # Scale positions to dimensionless vectors with approximately unit variance
        r_noisy = self.scale_inputs(noisy_atoms, timesteps)

        # Sequence local atom attention and aggregation to coarse-grained tokens
        atom_encoder_output = self.atom_attention_encoder(
            features=features,
            n_tokens=n_tokens,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noisy_pos=r_noisy
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
            token_repr=token_single,
            atom_single_skip_repr=atom_encoder_output.atom_single_skip_repr,  # (bs, n_atoms, c_atom)
            atom_single_skip_proj=atom_encoder_output.atom_single_skip_proj,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr=atom_encoder_output.atom_pair_skip_repr,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx=features["atom_to_token"],  # (bs, n_atoms)
            mask=atom_mask,  # (bs, n_atoms)
        )  # (bs, n_atoms, 3)

        # Rescale updates to positions and combine with input positions
        output_pos = self.rescale_with_updates(atom_pos_updates, noisy_atoms, timesteps)
        return output_pos

    def sample(
            self,
            features: Dict[str, Tensor],  # input feature dict
            s_inputs: Tensor,  # (bs, n_tokens, c_token)
            s_trunk: Tensor,  # (bs, n_tokens, c_token)
            z_trunk: Tensor,  # (bs, n_tokens, n_tokens, c_token)
            noise_schedule: Tensor,  # (n_steps, 1)
            samples_per_trunk: int = 32,
            gamma_0: float = 0.8,
            gamma_min: float = 1.0,
            noise_scale: float = 1.003,
            step_scale: float = 1.5,
            use_deepspeed_evo_attention: bool = False
    ) -> Vec3Array:
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
                        Residue number in the token’s original x chain.
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
            noise_schedule:
                [*, n_steps] noise schedule for the diffusion trajectory
            gamma_0:
                lower bound on the gamma. Defaults to value used in the paper.
            gamma_min:
                threshold for the lower bound on gamma. Defaults to value used in the paper.
            noise_scale:
                noise scale. Defaults to value used in the paper.
            step_scale:
                step scale. Defaults to value used in the paper.
            use_deepspeed_evo_attention:
                Whether to use Deepspeed's optimized kernel for attention pair bias
        Returns:
            Vec3Array object of shape (bs * samples_per_trunk, n_atoms) containing the sampled
            structures
        """
        # Grab data about the input
        batch_size, n_atoms, _ = features["ref_pos"].shape
        dtype, device = s_inputs.dtype, s_inputs.device
        n_steps, _ = noise_schedule.shape

        # Expand the batch by samples_per_trunk
        expand_batch = lambda t: t.repeat_interleave(samples_per_trunk, dim=0)
        feats = tensor_tree_map(expand_batch, features)
        s_inputs = expand_batch(s_inputs)
        s_trunk = expand_batch(s_trunk)
        z_trunk = expand_batch(z_trunk)

        # Sample random noise as the initial structure
        x_l = Vec3Array.randn((batch_size * samples_per_trunk, n_atoms), device)  # float32

        # Type cast noise schedule to float32 to prevent numerical issues in sampling
        noise_schedule = noise_schedule.to(x_l.x.dtype)

        for i in range(1, n_steps):
            # Centre random augmentation
            x_l = centre_random_augmentation(x_l)

            c_step = noise_schedule[i]
            prev_step = noise_schedule[i - 1]
            gamma = gamma_0 if c_step > gamma_min else 0.0

            # Expand c_step and prev_step for proper broadcasting
            c_step = c_step.unsqueeze(0).expand(batch_size * samples_per_trunk, 1)  # (bs * samples_per_trunk, 1)
            prev_step = prev_step.unsqueeze(0).expand(batch_size * samples_per_trunk, 1)  # (bs * samples_per_trunk, 1)

            t_hat = torch.mul(prev_step, torch.add(gamma, 1.0))
            normal_noise = Vec3Array.randn((batch_size * samples_per_trunk, n_atoms), device)
            zeta = noise_scale * torch.sqrt((t_hat ** 2 - prev_step ** 2)) * normal_noise
            x_noisy = x_l + zeta

            # Run DiffusionModule to denoise structure
            x_denoised = self.forward(
                noisy_atoms=x_noisy.to_tensor().to(dtype),  # revert to model dtype
                timesteps=t_hat.to(dtype),
                features=feats,
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

        return x_l
