import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.models.components.invariant_point_attention import InvariantPointAttention
from src.models.components.structure_transition import StructureTransition
from src.models.components.backbone_update import BackboneUpdate
from src.utils.rigid_utils import Rigids
from typing import Tuple

# TODO: this should be close to the StructureModule in its implementation


class StructureLayer(nn.Module):

    def __init__(self,
                 c_s,
                 c_z,
                 c_hidden_ipa,
                 n_head,
                 n_qk_point,
                 n_v_point,
                 ipa_dropout,
                 n_structure_transition_layer,
                 structure_transition_dropout
                 ):
        """Initialize a Structure Layer.
        :param c_s:
            Single representation channel dimension
        :param c_z:
            Pair representation channel dimension
        :param c_hidden_ipa:
            Hidden IPA channel dimension
        :param n_head:
            Number of attention heads
        :param n_qk_point:
            Number of query/key points to generate
        :param n_v_point:
            Number of value points to generate
        :param ipa_dropout:
            IPA dropout rate
        :param n_structure_transition_layer:
            Number of structure transition layers
        :param structure_transition_dropout:
            structure transition dropout rate
        """
        super(StructureLayer, self).__init__()

        self.c_s = c_s
        self.c_z = c_z

        self.ipa = InvariantPointAttention(
            c_s,
            c_z,
            c_hidden_ipa,
            n_head,
            n_qk_point,
            n_v_point
        )
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s,
            n_structure_transition_layer,
            structure_transition_dropout
        )

        # backbone update  TODO: it might be useful to zero the gradients on rotations.
        self.bb_update = BackboneUpdate(c_s)

    def forward(self, inputs) -> Tuple:
        """Updates a structure by explicitly attending the 3D frames."""
        s, z, t, mask = inputs
        s = s + self.ipa(s, z, t.to_nanometers(), mask)  # IPA requires nanometer units
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = checkpoint(self.transition, s)
        t = t.compose(self.bb_update(s).to_angstroms())  # predict in nanometers, compose in angstroms
        return s, z, t, mask

    # def forward(self, inputs) -> Tuple:
    # """Forward pass with gradient checkpointing."""
    # return checkpoint(self.forward_pass, inputs)


class StructureNet(nn.Module):

    def __init__(self,
                 c_s: int,
                 c_z: int,
                 n_structure_layer: int = 4,
                 n_structure_block: int = 1,
                 c_hidden_ipa: int = 16,
                 n_head_ipa: int = 12,
                 n_qk_point: int = 4,
                 n_v_point: int = 8,
                 ipa_dropout: float = 0.1,
                 n_structure_transition_layer: int = 1,
                 structure_transition_dropout: float = 0.1,
                 ):
        """Initializes a structure network.
        :param c_s:
            Single representation channel dimension
        :param c_z:
            Pair representation channel dimension
        :param n_structure_layer:
            Number of structure layers
        :param c_hidden_ipa:
            Hidden IPA channel dimension (multiplied by the number of heads)
        :param n_head_ipa:
            Number of attention heads in the IPA
        :param n_qk_point:
            Number of query/key points to generate
        :param n_v_point:
            Number of value points to generate
        :param ipa_dropout:
            IPA dropout rate
        :param n_structure_transition_layer:
            Number of structure transition layers
        :param structure_transition_dropout:
            structure transition dropout rate
        """
        super(StructureNet, self).__init__()

        self.n_structure_block = n_structure_block

        self.c_s = c_s
        self.c_z = c_z
        self.n_structure_layer = n_structure_layer

        # Initial projection and layer norms
        self.pair_rep_layer_norm = nn.LayerNorm(c_z)
        self.single_rep_layer_norm = nn.LayerNorm(c_s)
        self.single_rep_linear = nn.Linear(c_s, c_s)

        layers = [
            StructureLayer(
                c_s, c_z,
                c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout,
                n_structure_transition_layer, structure_transition_dropout
            )
            for _ in range(n_structure_layer)
        ]
        self.net = nn.Sequential(*layers)

    def forward(
            self,
            single_rep: torch.Tensor,
            pair_rep: torch.Tensor,
            transforms: Rigids,
            mask: torch.Tensor = None
    ) -> Rigids:
        """Applies the structure module on the current transforms given single and pair representations.

        :param single_rep:
            [*, N_res, C_s] single representation
        :param pair_rep:
            [*, N_res, N_res, C_z] pair representation
        :param transforms:
            [*, N_res] transformation object
        :param mask:
            [*, N_res] mask

        :returns
            [*, N_res] updated transforms
        """

        # Initial projection and layer norms
        single_rep = self.single_rep_layer_norm(single_rep)
        single_rep = self.single_rep_linear(single_rep)
        pair_rep = self.pair_rep_layer_norm(pair_rep)

        # Initial structure
        structure = (single_rep, pair_rep, transforms, mask)

        # Updates with shared weights
        for _ in range(self.n_structure_block):
            # structure = checkpoint_sequential(self.net,
            #                                  segments=self.n_structure_layer,
            #                                  x=structure)
            structure = self.net(structure)

        # Return updated transforms
        return structure[2]
