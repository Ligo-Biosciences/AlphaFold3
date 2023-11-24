import torch
from torch import nn


class PairFeatureNet(nn.Module):
	"""Computes pair representation features."""

	def __init__(
			self,
			c_z: int,
			relpos_k: int = 32
	):
		"""Initializes PairFeatureNet.
		:param c_z:
			number of channels in the pair representation
		:param relpos_k:
			cap at the number of residues to include in positional encoding in sequence space.
			Defaults to 32 as in AlphaFold.
		"""
		super(PairFeatureNet, self).__init__()
		assert c_z % 2 == 0, "Channels in pair representation c_z should be divisible by 2."

		self.c_p = c_z
		self.relpos_k = relpos_k
		self.n_bin = 2 * relpos_k + 1

		# Relative positional and distance embeddings
		self.linear_relpos = nn.Linear(self.n_bin, c_z // 2)
		self.linear_reldist = nn.Linear(10, c_z // 2)  # 10 bins, AlphaFold-style

	def one_hot(self, x, v_bins):
		"""One-hot encoding with the nearest bin.
		Implements Alg. 5 in Jumper et al. 2021.
		:param x:
			[...] a tensor of floats
		:param v_bins:
			[n_bins] a tensor of floats containing bin values
		"""
		b = torch.argmin(torch.abs(x[..., None] - v_bins[None, ...]), dim=-1)
		one_hot = nn.functional.one_hot(b, num_classes=len(v_bins))
		return one_hot.float()

	def relpos(self, residue_idx: torch.Tensor) -> torch.Tensor:
		"""Implements Alg. 4 in Supp. Info. Jumper et al. 2021.

		:param residue_idx:
			[*, n_res] a tensor encoding the residue indices in sequence space

		:return:
			[*, n_res, n_res, c_p] tensor of embedded positional encodings
		"""
		# Compute relative positional differences [b, n_res, n_res]
		d = residue_idx[:, :, None] - residue_idx[:, None, :]

		# [n_bin]
		v_bins = torch.arange(-self.relpos_k, self.relpos_k + 1).to(residue_idx.device)

		# One-hot encode to the nearest bin
		one_hot = self.one_hot(d, v_bins)

		# Embed [b, n_res, n_res, c_p]
		positional_pair_embed = self.linear_relpos(one_hot)
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
		v_bins = torch.linspace(3.375, 21.375, steps=10)  # same bin values as in AlphaFold
		one_hot = self.one_hot(d_ij, v_bins)
		distance_pair_embed = self.linear_reldist(one_hot)
		return distance_pair_embed

	def residue_mask2pair_mask(self, residue_mask: torch.Tensor):
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
			ca_coordinates: torch.Tensor,
			residue_mask: torch.Tensor
	):
		"""Featurizes a protein given Ca coordinates and residue indices.
		:param residue_idx:
			[*, n_res] tensor encoding the residue indices in the original
		:param ca_coordinates:
			[*, n_res, 3] tensor containing the carbon alpha coordinates
		:param residue_mask:
			[*, n_res] tensor of residue mask (0 where coordinates are missing, 1 otherwise)
		"""
		# Compute relpos features
		relpos_features = self.relpos(residue_idx)

		# Compute reldist features
		reldist_features = self.reldist(ca_coordinates)

		# Mask missing residue distances with pair mask
		pair_mask = self.residue_mask2pair_mask(residue_mask)
		reldist_features *= pair_mask.unsqueeze(-1)

		# Concatenate
		pair_features = torch.concat([relpos_features, reldist_features], dim=-1)
		return pair_features
