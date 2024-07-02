"""Construct an initial 1D embedding."""
import torch
from torch import Tensor
from torch import nn
from src.models.components.atom_attention import AtomAttentionEncoder
from typing import Dict, NamedTuple, Tuple
from src.models.components.primitives import LinearNoBias
from src.models.components.relative_position_encoding import RelativePositionEncoding
from src.utils.checkpointing import get_checkpoint_fn
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
            n_tokens: int,
            c_token: int = 384,
            c_atom: int = 128,
            c_atompair: int = 16,
            c_trunk_pair: int = 16,
            num_blocks: int = 3,
            num_heads: int = 4,
            dropout=0.0,
            n_queries: int = 32,
            n_keys: int = 128,
            device=None,
            dtype=None,
    ):
        super().__init__()
        self.n_tokens = n_tokens
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

        # Atom Attention encoder
        self.encoder = AtomAttentionEncoder(
            n_tokens=self.n_tokens,
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_trunk_pair=self.c_trunk_pair,
            no_blocks=self.num_blocks,
            no_heads=self.num_heads,
            dropout=self.dropout,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            trunk_conditioning=False,  # no trunk conditioning for the x feature embedder
        )

    def forward(
            self,
            features: Dict[str, torch.Tensor],
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass of the x feature embedder.
        Args:
            features:
                Dictionary containing the x features:
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
            mask:
                [*, N_atoms] mask indicating which atoms are valid (non-padding).
        Returns:
            [*, N_tokens, c_token] Embedding of the x features.
        """
        # Encode the x features
        output = self.encoder(features=features, mask=mask)
        per_token_features = output.token_single  # f_restype, f_profile, and f_deletion_mean do not exist for design
        return per_token_features


class ProteusFeatures(NamedTuple):
    """Structured output class for Proteus features."""
    s_inputs: torch.Tensor  # (bs, n_tokens, c_token)
    s_trunk: torch.Tensor  # (bs, n_tokens, c_token)
    z_trunk: torch.Tensor  # (bs, n_tokens, n_tokens, c_token)


class ProteusFeatureEmbedder(nn.Module):
    """Convenience class for the Proteus experiment."""
    def __init__(
            self,
            n_tokens: int,
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
        self.n_tokens = n_tokens
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
            n_tokens=n_tokens,
            num_blocks=num_blocks,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_trunk_pair=c_trunk_pair,
            num_heads=num_heads,
            dropout=dropout,
            n_queries=n_queries,
            n_keys=n_keys,
            device=device,
            dtype=dtype
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
                Dictionary containing the x features
            atom_mask:
                [*, N_atoms] mask indicating which atoms are valid (non-padding).
            token_mask:
                [*, N_tokens] mask indicating which tokens are valid (non-padding).
        Returns:
            [*, N_tokens, c_token] Embedding of the x features.
        """
        # Encode the x features
        per_token_features = self.input_feature_embedder(features=features, mask=atom_mask)
        # f_restype, f_profile, and f_deletion_mean do not exist for design

        # Compute s_trunk
        s_trunk = self.linear_s_init(per_token_features)

        # Compute z_trunk
        z_trunk = self.linear_z_col(per_token_features[:, :, None, :]) +\
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
                    Dictionary containing the x features
                atom_mask:
                    [*, N_atoms] mask indicating which atoms are valid (non-padding).
                token_mask:
                    [*, N_tokens] mask indicating which tokens are valid (non-padding).
            Returns:
                [*, N_tokens, c_token] Embedding of the x features.
        """
        return checkpoint(self._forward, features, atom_mask, token_mask)

