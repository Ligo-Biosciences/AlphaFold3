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
from typing import Dict, Tuple
from src.utils.geometry.vector import Vec3Array
from src.diffusion.conditioning import DiffusionConditioning
from src.diffusion.attention import DiffusionTransformer
from src.models.components.atom_attention import AtomAttentionEncoder, AtomAttentionDecoder
from src.models.components.primitives import Linear


class DiffusionModule(torch.nn.Module):
    def __init__(
            self,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_token: int = 768,
            c_tokenpair: int = 128,
            n_tokens: int = 384,
            atom_encoder_blocks: int = 3,
            atom_encoder_heads: int = 16,
            dropout: float = 0.0,
            atom_attention_n_queries: int = 32,
            atom_attention_n_keys: int = 128,
            atom_decoder_blocks: int = 3,
            atom_decoder_heads: int = 16,
            token_transformer_blocks: int = 24,
            token_transformer_heads: int = 16,
    ):
        super(DiffusionModule, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_tokenpair = c_tokenpair
        self.n_tokens = n_tokens
        self.atom_encoder_blocks = atom_encoder_blocks
        self.atom_encoder_heads = atom_encoder_heads
        self.dropout = dropout
        self.atom_attention_n_queries = atom_attention_n_queries
        self.atom_attention_n_keys = atom_attention_n_keys
        self.token_transformer_blocks = token_transformer_blocks
        self.token_transformer_heads = token_transformer_heads

        # Conditioning
        self.diffusion_conditioning = DiffusionConditioning(c_token=c_token, c_pair=c_tokenpair)

        # Sequence-local atom attention and aggregation to coarse-grained tokens
        self.atom_attention_encoder = AtomAttentionEncoder(
            n_tokens=n_tokens,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_trunk_pair=c_tokenpair,
            num_blocks=atom_decoder_blocks,
            num_heads=atom_encoder_heads,
            dropout=dropout,
            n_queries=atom_attention_n_queries,
            n_keys=atom_attention_n_keys,
            trunk_conditioning=True
        )

        # Full self-attention on token level
        self.linear_token_residual = Linear(c_token, c_token, bias=False, init='final')
        self.token_residual_layer_norm = nn.LayerNorm(c_token)
        self.diffusion_transformer = DiffusionTransformer(
            c_token=c_token,
            c_pair=c_tokenpair,
            num_blocks=token_transformer_blocks,
            num_heads=token_transformer_heads,
            dropout=dropout,
        )
        self.token_post_layer_norm = nn.LayerNorm(c_token)

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

    def forward(
            self,
            noisy_atoms: Vec3Array,  # (bs, n_atoms)
            timesteps: torch.Tensor,  # (bs, 1)
            features: Dict[str, torch.Tensor],  # input feature dict
            s_inputs: torch.Tensor,  # (bs, n_tokens, c_token)
            s_trunk: torch.Tensor,  # (bs, n_tokens, c_token)
            z_trunk: torch.Tensor,  # (bs, n_tokens, n_tokens, c_pair)
            sd_data: float = 16.0,
            token_mask: torch.Tensor = None,  # (bs, n_tokens)
            atom_mask: torch.Tensor = None  # (bs, n_atoms)
    ) -> Vec3Array:
        """Diffusion module that denoises atomic coordinates based on conditioning.
        Args:
            noisy_atoms:
                vector of noisy atom positions (bs, n_atoms)
            timesteps:
                tensor of timesteps (bs, 1)
            features:
                input feature dictionary containing the tensors:
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
                    "residue_index":
                        [*, N_tokens] Residue number in the token’s original input chain.
                    "token_index":
                        [*, N_tokens] Token number. Increases monotonically; does not restart at 1
                        for new chains.
                    "asym_id":
                        [*, N_tokens] Unique integer for each distinct chain.
                    "entity_id":
                        [*, N_tokens] Unique integer for each distinct sequence.
                    "sym_id":
                        [*, N_tokens] Unique integer within chains of this sequence. E.g. if chains
                        A, B and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0]

            s_inputs:
                [*, n_tokens, c_token] Single conditioning input
            s_trunk:
                [*, n_tokens, c_token] Single conditioning from Pairformer trunk
            z_trunk:
                [*, n_tokens, n_tokens, c_pair] Pair conditioning from Pairformer trunk
            sd_data:
                Scaling factor for the timesteps before fourier embedding
            token_mask:
                [*, N_tokens] binary mask for tokens, whether token is present (not padding)
            atom_mask:
                [*, N_atoms] binary mask for atoms, whether atom is present (will still be 1.0 if
                atom is missing from the crystal structure, only 0.0 for padding)

        """
        # Conditioning
        token_repr, pair_repr = self.diffusion_conditioning(timesteps=timesteps,
                                                            features=features,
                                                            s_inputs=s_inputs,
                                                            s_trunk=s_trunk,
                                                            z_trunk=z_trunk,
                                                            mask=token_mask)

        # Scale positions to dimensionless vectors with approximately unit variance
        scale_factor = torch.reciprocal((torch.sqrt(torch.add(timesteps ** 2, sd_data ** 2))))
        r_noisy = noisy_atoms / scale_factor

        # Sequence local atom attention and aggregation to coarse-grained tokens
        atom_encoder_output = self.atom_attention_encoder(features, s_trunk, z_trunk, r_noisy)

        # Full self-attention on token level
        token_single = atom_encoder_output.token_single + self.linear_token_residual(
            self.token_residual_layer_norm(token_repr)
        )
        token_single = self.diffusion_transformer(single_repr=token_single,
                                                  single_proj=token_repr,
                                                  pair_repr=pair_repr,
                                                  mask=token_mask)
        token_single = self.token_post_layer_norm(token_single)

        # Broadcast token activations to atoms and run sequence-local atom attention
        atom_pos_updates = self.atom_attention_decoder(
            token_repr=token_single,
            atom_single_skip_repr=atom_encoder_output.atom_single_skip_repr,  # (bs, n_atoms, c_atom)
            atom_single_skip_proj=atom_encoder_output.atom_single_skip_proj,  # (bs, n_atoms, c_atom)
            atom_pair_skip_repr=atom_encoder_output.atom_pair_skip_repr,  # (bs, n_atoms, n_atoms, c_atom)
            tok_idx=features["atom_to_token"],  # (bs, n_atoms)
            mask=atom_mask  # (bs, n_atoms)
        )

        # Rescale updates to positions and combine with input positions
        common = torch.add(timesteps ** 2, sd_data ** 2)
        noisy_pos_scale = torch.div(sd_data ** 2, common)  # (bs, 1)
        atom_pos_update_scale = torch.div(torch.mul(sd_data, timesteps), torch.sqrt(common))
        output_pos = noisy_atoms * noisy_pos_scale + atom_pos_updates * atom_pos_update_scale
        return output_pos
