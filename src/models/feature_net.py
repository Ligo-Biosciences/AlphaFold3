import torch
from torch import nn
from src.utils.tensor_utils import one_hot
from src.utils.geometry.rigid_matrix_vector import Rigid3Array
from src.utils.geometry.vector import Vec3Array
from src.models.components.invariant_point_attention import InvariantPointAttention
from src.models.components.primitives import generate_sinusoidal_encodings


class FeatureNet(nn.Module):
    """Computes single and pair representation features."""

    def __init__(
            self,
            c_z: int,
            c_s: int,
            c_ipa: int,
            no_heads_ipa: int,
            no_qk_points: int,
            no_v_points: int,
            relpos_k: int = 32
    ):
        """Initializes FeatureNet.
        Args:
            c_z:
                number of channels in the pair representation
            c_s:
                a
            c_ipa:
                a
            no_heads_ipa:
                a
            no_qk_points:
                a
            no_v_points:
                a
            relpos_k:
                cap at the number of residues to include in positional encoding in sequence space.
                Defaults to 32 as in AlphaFold.
        """

        super(FeatureNet, self).__init__()
        assert c_z % 2 == 0, "Channels in pair representation c_hidden should be divisible by 2."
        self.c_s = c_s
        self.c_z = c_z
        self.relpos_k = relpos_k
        self.n_bin = 2 * relpos_k + 1

        # Relative positional and distance embeddings
        self.linear_relpos = nn.Linear(self.n_bin, c_z // 2)
        self.linear_reldist = nn.Linear(15, c_z // 2)  # 15 bins, AlphaFold-style
        self.pair_layer_norm = nn.LayerNorm(self.c_z)

        # Single representation
        self.ipa = InvariantPointAttention(c_s=c_s,
                                           c_z=c_z,
                                           c_hidden=c_ipa,
                                           no_heads=no_heads_ipa,
                                           no_qk_points=no_qk_points,
                                           no_v_points=no_v_points)
        self.single_layer_norm = nn.LayerNorm(self.c_s)

    def relpos(self, residue_idx: torch.Tensor) -> torch.Tensor:
        """Implements Alg. 4 in Supp. Info. Jumper et al. 2021.

        :param residue_idx:
            [*, n_res] a tensor encoding the residue indices in sequence space

        :return:
            [*, n_res, n_res, c_z] tensor of embedded positional encodings
        """
        # Compute relative positional differences [b, n_res, n_res]
        d = residue_idx[:, :, None] - residue_idx[:, None, :]

        # [n_bin]
        v_bins = torch.arange(-self.relpos_k, self.relpos_k + 1).to(residue_idx)

        # One-hot encode to the nearest bin
        one_hot_rep = one_hot(d, v_bins)

        # Embed [b, n_res, n_res, c_z]
        positional_pair_embed = self.linear_relpos(one_hot_rep)
        return positional_pair_embed

    def reldist(self, ca_coordinates: torch.Tensor) -> torch.Tensor:
        """Implements part of Alg. 32 in Supp. Info. Jumper et al. 2021.
        Computes relative distances and encodes them in bins between 3-22 Angstroms.
        :param ca_coordinates:
            [*, n_res, 3] tensor containing the coordinates of the Ca atoms in a protein
            This is equivalent to the translation vectors in a Rigids object.
        :returns a pair representation with embedded pair distances of backbone atoms
        """
        d_ij = torch.sum((ca_coordinates[:, :, None, :] - ca_coordinates[:, None, :, :]) ** 2, dim=-1)
        v_bins = torch.linspace(3.375, 21.375, steps=15).to(ca_coordinates)  # same bin values as in AlphaFold
        one_hot_rep = one_hot(d_ij, v_bins)
        distance_pair_embed = self.linear_reldist(one_hot_rep)
        return distance_pair_embed

    def residue_to_pair_mask(self, residue_mask: torch.Tensor):
        """Converts a residue mask to pair mask.
        :param residue_mask:
            [*, n_res] tensor of residue mask (0 where coordinates are missing, 1 otherwise)
        :return:
            pair_mask of shape [*, n_res, n_res]
        """
        pair_mask = residue_mask[:, :, None] * residue_mask[:, None, :]
        return pair_mask

    def forward(
            self,
            residue_idx: torch.Tensor,
            coordinates: torch.Tensor,
            residue_mask: torch.Tensor
    ):
        """Featurizes a protein given Ca coordinates and residue indices.
        Args:
            residue_idx:
                [*, n_res] tensor encoding the residue indices in the original
            coordinates:
                [*, n_res, 4, 3] tensor containing the backbone coordinates
            residue_mask:
                [*, n_res] tensor of residue mask (0 where coordinates are missing, 1 otherwise)
        """
        # Compute relpos features
        relpos_features = self.relpos(residue_idx)

        # Compute reldist features
        reldist_features = self.reldist(coordinates[:, :, 1, :])

        # Mask missing residue distances with pair mask
        pair_mask = self.residue_to_pair_mask(residue_mask)
        reldist_features *= pair_mask.unsqueeze(-1)

        # Concatenate
        pair_features = torch.concat([relpos_features, reldist_features], dim=-1)
        pair_features = self.pair_layer_norm(pair_features)

        # Single rep features as residue index  [*, n_res, c_s] on top of IPA
        single_repr = generate_sinusoidal_encodings(residue_idx, c_s=self.c_s)
        gt_frames = Rigid3Array.from_3_points(Vec3Array.from_array(coordinates[:, :, 0, :]),  # N
                                              Vec3Array.from_array(coordinates[:, :, 1, :]),  # CA
                                              Vec3Array.from_array(coordinates[:, :, 2, :]))  # C
        single_repr = single_repr + self.ipa(s=single_repr, z=pair_features, r=gt_frames, mask=residue_mask)
        single_repr = self.single_layer_norm(single_repr)
        return {"single": single_repr, "pair": pair_features}
