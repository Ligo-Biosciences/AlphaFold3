from functools import reduce
import importlib
import math
import sys
from operator import mul

import torch
import torch.nn as nn
from typing import Optional, Tuple, Sequence, Union

from src.models.components.primitives import Linear, LayerNorm
from src.models.components.backbone_update import BackboneUpdate
from src.models.components.invariant_point_attention import InvariantPointAttention
from src.models.components.structure_transition import StructureTransition

from src.utils.geometry.rigid_matrix_vector import Rigid3Array
from src.common.residue_constants import restype_atom14_rigid_group_positions
from src.utils.tensor_utils import dict_multimap


class StructureModule(nn.Module):
    def __init__(
            self,
            c_s,
            c_z,
            c_ipa,
            c_resnet,
            no_heads_ipa,
            no_qk_points,
            no_v_points,
            dropout_rate,
            no_blocks,
            no_transition_layers,
            no_resnet_blocks,
            no_angles,
            trans_scale_factor,
            epsilon,
            inf,
            **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModule, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        ipa = InvariantPointAttention
        self.ipa = ipa(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        self.bb_update = BackboneUpdate(self.c_s, full_quat=False)

    def _forward_multimer(
            self,
            evoformer_output_dict,
            aatype,
            mask=None,
    ):
        s = evoformer_output_dict["single"]  # the single representation

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])

        # [*, N, C_s]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        # [*, N, C_s]
        s = self.linear_in(s)

        # [*, N]
        rigids = Rigid3Array.identity(
            s.shape[:-1],
            s.device,
        )
        outputs = []
        for i in range(self.no_blocks):
            # [*, N, C_s]
            s = s + self.ipa(s, z, rigids, mask)  # TODO: the coordinates should be in nanometers, not sure here
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # [*, N]
            rigids = rigids @ self.bb_update(s)  # compose

            # Scale frame translations
            rigids = rigids.scale_translation(self.trans_scale_factor)

            # Convert to atom positions
            pred_xyz = self.frames_to_atom4_pos(rigids)

            preds = {
                "frames": rigids.to_tensor(),
                "positions": pred_xyz,
            }

            preds = {k: v.to(dtype=s.dtype) for k, v in preds.items()}

            outputs.append(preds)

            rigids = rigids.stop_rot_gradient()

        del z

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s

        return outputs

    def forward(
            self,
            evoformer_output_dict,
            aatype,
            mask=None,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        outputs = self._forward_multimer(evoformer_output_dict, aatype, mask)
        return outputs

    def _init_residue_constants(self, float_dtype, device):
        """Initialize the literature positions on the correct device."""
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def frames_to_atom4_pos(
            self,
            frames: Rigid3Array,
            reference_atom: str = "ALA",
    ):
        """Given backbone frames, convert to atom positions using the literature positions
        of the reference atom.
        Args:
            frames: the backbone frames
            reference_atom: the atom name to use as the reference in 3-letter code
        """
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(frames.dtype, frames.device)
        raise NotImplementedError("Implement me!")
