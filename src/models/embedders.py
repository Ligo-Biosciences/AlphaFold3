"""Construct an initial 1D embedding."""
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from src.models.components.atom_attention import AtomAttentionEncoder
from typing import Dict, NamedTuple, Tuple, Optional
from src.models.components.primitives import LinearNoBias, LayerNorm, Linear
from src.models.components.relative_position_encoding import RelativePositionEncoding
from src.models.template import TemplatePairStack
from src.utils.tensor_utils import add
from src.utils.checkpointing import get_checkpoint_fn
from src.utils.geometry.vector import Vec3Array

checkpoint = get_checkpoint_fn()


class InputFeatureEmbedder(nn.Module):
    """A class that performs attention over all atoms in order to encode the information
    about the chemical structure of all the molecules, leading to a single representation
    representing all the tokens.
    - Embed per-atom features
    - Concatenate the per-token features
    """

    def __init__(
            self,
            c_token: int = 384,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 128,
            num_blocks: int = 3,
            num_heads: int = 4,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_trunk_pair = c_trunk_pair
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys

        # Atom Attention encoder
        self.encoder = AtomAttentionEncoder(
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_trunk_pair=self.c_trunk_pair,
            no_blocks=self.num_blocks,
            no_heads=self.num_heads,
            dropout=self.dropout,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            trunk_conditioning=False,  # no trunk conditioning for the input feature embedder
        )

        # Output projection
        self.restype_embedding = LinearNoBias(21, c_token)  # for the restype embedding
        self.output_ln = LayerNorm(c_token)

    def forward(
            self,
            features: Dict[str, Tensor],
            n_tokens: int,
            mask: Tensor = None,
            use_flash: bool = False,
    ) -> Tensor:
        """Forward pass of the input feature embedder.
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
                    "aatype":
                        [*, N_token] Amino acid type for each token.
                    "atom_to_token":
                        [*, N_atoms] Token index for each atom in the flat atom representation.
            n_tokens:
                number of tokens
            mask:
                [*, N_atoms] mask indicating which atoms are valid (non-padding).
            use_flash:
                Whether to use Flash attention within AtomAttentionEncoder.
        Returns:
            [*, N_tokens, c_token] Embedding of the input features.
        """
        # Encode the input features
        output = self.encoder(features=features, mask=mask, n_tokens=n_tokens, use_flash=use_flash)
        per_token_features = output.token_single  # TODO: add f_profile, and f_deletion_mean
        f_restype = F.one_hot(features["aatype"], num_classes=21).to(per_token_features.dtype)
        per_token_features = per_token_features + self.restype_embedding(f_restype)
        per_token_features = self.output_ln(per_token_features)
        return per_token_features


class InputEmbedder(nn.Module):
    """Input embedder for AlphaFold3 that initializes the single and pair representations."""

    def __init__(
            self,
            c_token: int = 384,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 128,
    ):
        super(InputEmbedder, self).__init__()

        # InputFeatureEmbedder for the s_inputs representation
        self.input_feature_embedder = InputFeatureEmbedder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_trunk_pair=c_trunk_pair
        )

        # Projections
        self.linear_single = LinearNoBias(c_token, c_token)
        self.linear_proj_i = LinearNoBias(c_token, c_trunk_pair)
        self.linear_proj_j = LinearNoBias(c_token, c_trunk_pair)
        self.linear_bonds = LinearNoBias(1, c_trunk_pair)

        # Relative position encoding
        self.relpos = RelativePositionEncoding(c_pair=c_trunk_pair)

    def forward(
            self,
            features: Dict[str, Tensor],
            inplace_safe: bool = False,
            use_flash: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            features:
                Dictionary containing the following input features:
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
                    "atom_mask" ([*, n_atoms]):
                        mask indicating which atoms are valid (non-padding).
                    "token_mask" ([*, N_token]):
                        mask indicating which tokens are valid (non-padding).
            inplace_safe:
                whether to use inplace operations
            use_flash:
                whether to use Flash attention within AtomAttentionEncoder.
        """
        *_, n_tokens, _ = features["token_bonds"].shape

        # Extract masks
        atom_mask = features["atom_mask"]
        token_mask = features["token_mask"]

        # Single representation with input feature embedder
        s_inputs = self.input_feature_embedder(
            features,
            n_tokens=n_tokens,
            mask=atom_mask,
            use_flash=use_flash
        )

        # Projections
        s_init = self.linear_single(s_inputs)
        z_init = add(
            self.linear_proj_i(s_inputs[..., None, :]),
            self.linear_proj_j(s_inputs[..., None, :, :]),
            inplace=inplace_safe
        )  # (*, n_tokens, n_tokens, c_trunk_pair)

        # Add relative position encoding
        z_init = add(
            z_init,
            self.relpos(features, mask=token_mask),
            inplace=inplace_safe
        )

        # Add token bond information
        # z_init = add(
        #    z_init,
        #    self.linear_bonds(features["token_bonds"][..., None]),
        #    inplace=inplace_safe
        # )
        return s_inputs, s_init, z_init


class TemplateEmbedder(nn.Module):
    def __init__(
            self,
            no_blocks: int = 2,
            c_template: int = 32,
            c_z: int = 128,
            clear_cache_between_blocks: bool = False
    ):
        super(TemplateEmbedder, self).__init__()

        self.proj_pair = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, c_template)
        )
        no_template_features = 108
        self.linear_templ_feat = LinearNoBias(no_template_features, c_template)
        self.pair_stack = TemplatePairStack(
            no_blocks=no_blocks,
            c_template=c_template,
            clear_cache_between_blocks=clear_cache_between_blocks
        )
        self.v_to_u_ln = LayerNorm(c_template)
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(c_template, c_template)
        )
        self.clear_cache_between_blocks = clear_cache_between_blocks

    def forward(
            self,
            features: Dict[str, Tensor],
            z_trunk: Tensor,
            pair_mask: Tensor,
            chunk_size: Optional[int] = None,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ) -> Tensor:
        """
        TODO: modify this function to take the same features as the OpenFold template embedder. That will allow
         minimal changes in the data pipeline.
        Args:
            features:
                Dictionary containing the template features:
                    "template_restype":
                        [*, N_templ, N_token, 32] One-hot encoding of the template sequence.
                    "template_pseudo_beta_mask":
                        [*, N_templ, N_token] Mask indicating if the Cβ (Cα for glycine)
                        has coordinates for the template at this residue.
                    "template_backbone_frame_mask":
                        [*, N_templ, N_token] Mask indicating if coordinates exist for all
                        atoms required to compute the backbone frame (used in the template_unit_vector feature).
                    "template_distogram":
                        [*, N_templ, N_token, N_token, 39] A one-hot pairwise feature indicating the distance
                        between Cβ atoms (Cα for glycine). Pairwise distances are discretized into 38 bins of
                        equal width between 3.25 Å and 50.75 Å; one more bin contains any larger distances.
                    "template_unit_vector":
                        [*, N_templ, N_token, N_token, 3] The unit vector of the displacement of the Cα atom of
                        all residues within the local frame of each residue.
                    "asym_id":
                        [*, N_token] Unique integer for each distinct chain.
            z_trunk:
                [*, N_token, N_token, c_z] pair representation from the trunk.
            pair_mask:
                [*, N_token, N_token] mask indicating which pairs are valid (non-padding).
            chunk_size:
                Chunk size for the pair stack.
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo attention within the pair stack.
            use_lma:
                Whether to use LMA within the pair stack.
            inplace_safe:
                Whether to use inplace operations.
        """
        # Grab data about the inputs
        *bs, n_templ, n_token, _ = features["template_restype"].shape
        bs = tuple(bs)

        # Compute masks
        b_frame_mask = features["template_backbone_frame_mask"]
        b_frame_mask = b_frame_mask[..., None] * b_frame_mask[..., None, :]  # [*, n_templ, n_token, n_token]
        b_pseudo_beta_mask = features["template_pseudo_beta_mask"]
        b_pseudo_beta_mask = b_pseudo_beta_mask[..., None] * b_pseudo_beta_mask[..., None, :]

        template_feat = torch.cat([
            features["template_distogram"],
            b_frame_mask[..., None],  # [*, n_templ, n_token, n_token, 1]
            features["template_unit_vector"],
            b_pseudo_beta_mask[..., None]
        ], dim=-1)

        # Mask out features that are not in the same chain
        asym_id_i = features["asym_id"][..., None, :].expand((bs + (n_templ, n_token, n_token)))
        asym_id_j = features["asym_id"][..., None].expand((bs + (n_templ, n_token, n_token)))
        same_asym_id = torch.isclose(asym_id_i, asym_id_j).to(template_feat.dtype)
        template_feat = template_feat * same_asym_id.unsqueeze(-1)

        # Add residue type information
        temp_restype_i = features["template_restype"][..., None, :].expand(bs + (n_templ, n_token, n_token, -1))
        temp_restype_j = features["template_restype"][..., None, :, :].expand(bs + (n_templ, n_token, n_token, -1))
        template_feat = torch.cat([template_feat, temp_restype_i, temp_restype_j], dim=-1)

        # Run the pair stack per template
        single_templates = torch.unbind(template_feat, dim=-4)  # each element shape [*, n_token, n_token, c_template]
        z_proj = self.proj_pair(z_trunk)
        u = torch.zeros_like(z_proj)
        for t in range(len(single_templates)):
            # Project and add the template features
            v = z_proj + self.linear_templ_feat(single_templates[t])
            # Run the pair stack
            v = add(v,
                    self.pair_stack(v,
                                    pair_mask=pair_mask,
                                    chunk_size=chunk_size,
                                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                    use_lma=use_lma, inplace_safe=inplace_safe),
                    inplace=inplace_safe
                    )
            # Normalize and add to u
            u = add(u, self.v_to_u_ln(v), inplace=inplace_safe)
            del v
        u = torch.div(u, n_templ)  # average
        u = self.output_proj(u)
        return u


class TemplatePairEmbedder(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            c_dgram: int,
            c_aatype: int
    ):
        super(TemplatePairEmbedder, self).__init__()

        self.dgram_linear = Linear(c_dgram, c_out, init='relu')

    def forward(
            self,
            template_dgram: Tensor,
            aatype_one_hot: Tensor,
            query_embedding: Tensor,
            pseudo_beta_mask: Tensor,
            backbone_mask: Tensor,
            multichain_mask_2d: Tensor,
            unit_vector: Vec3Array,
    ) -> Tensor:
        # TODO: implement pair embedder
        pass


class TemplateEmbedderMultimer(nn.Module):
    """Template embedder used in AF3-Multimer, will replace the TemplateEmbedder above."""

    def __init__(self, config):
        super(TemplateEmbedderMultimer, self).__init__()
        self.config = config
        # pair embedder
        # template pair stack

    def forward(self):
        # TODO: integrate pair embedder with the template pair stack
        pass


class ProteusFeatures(NamedTuple):
    """Structured output class for Proteus features."""
    s_inputs: Tensor  # (bs, n_tokens, c_token)
    s_trunk: Tensor  # (bs, n_tokens, c_token)
    z_trunk: Tensor  # (bs, n_tokens, n_tokens, c_token)


class ProteusFeatureEmbedder(nn.Module):
    """Convenience class for the Proteus experiment."""

    def __init__(
            self,
            c_token: int = 384,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 4,
            dropout: float = 0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_trunk_pair = c_trunk_pair
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.device = device
        self.dtype = dtype

        self.input_feature_embedder = InputFeatureEmbedder(
            num_blocks=num_blocks,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_trunk_pair=c_trunk_pair,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys
        )
        self.linear_s_init = LinearNoBias(c_token, c_token)
        self.linear_z_col = LinearNoBias(c_token, c_trunk_pair)
        self.linear_z_row = LinearNoBias(c_token, c_trunk_pair)
        self.relative_pos_encoder = RelativePositionEncoding(c_trunk_pair)

    def _forward(
            self,
            features: Dict[str, torch.Tensor],
            atom_mask: torch.Tensor = None,
            token_mask: torch.Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Proteus feature embedder.
        Args:
            features:
                Dictionary containing the input features
            atom_mask:
                [*, N_atoms] mask indicating which atoms are valid (non-padding).
            token_mask:
                [*, N_tokens] mask indicating which tokens are valid (non-padding).
        Returns:
            [*, N_tokens, c_token] Embedding of the input features.
        """
        # Grab data about the input
        *_, n_tokens = features["token_index"].shape

        # Encode the input features
        per_token_features = self.input_feature_embedder(features=features, n_tokens=n_tokens, mask=atom_mask)
        # f_restype, f_profile, and f_deletion_mean do not exist for design

        # Compute s_trunk
        s_trunk = self.linear_s_init(per_token_features)

        # Compute z_trunk
        z_trunk = self.linear_z_col(per_token_features[:, :, None, :]) + \
                  self.linear_z_row(per_token_features[:, None, :, :])
        z_trunk = z_trunk + self.relative_pos_encoder(features, token_mask)

        return per_token_features, s_trunk, z_trunk

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            atom_mask: torch.Tensor = None,
            token_mask: torch.Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Proteus feature embedder.
            Args:
                features:
                    Dictionary containing the input features
                atom_mask:
                    [*, N_atoms] mask indicating which atoms are valid (non-padding).
                token_mask:
                    [*, N_tokens] mask indicating which tokens are valid (non-padding).
            Returns:
                [*, N_tokens, c_token] Embedding of the x features.
        """
        return checkpoint(self._forward, features, atom_mask, token_mask)
