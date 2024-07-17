"""AlphaFold3 model implementation."""
import torch
from torch import nn
from torch import Tensor
from src.models.embedders import InputEmbedder, TemplateEmbedder
from src.models.pairformer import PairformerStack
from src.models.msa_module import MSAModule
from src.models.diffusion_module import DiffusionModule
from src.models.heads import DistogramHead, ConfidenceHead
from src.models.components.primitives import LinearNoBias, LayerNorm
from src.utils.tensor_utils import add
from typing import Tuple, Dict


class AlphaFold3(nn.Module):
    def __init__(self, config):
        super(AlphaFold3, self).__init__()
        self.globals = config.globals
        self.config = config.model

        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"]
        )

        self.template_embedder = TemplateEmbedder(
            **self.config["template_embedder"]
        )

        self.msa_module = MSAModule(
            **self.config["msa_module"]
        )

        self.pairformer = PairformerStack(
            **self.config["pairformer_stack"]
        )

        self.diffusion_module = DiffusionModule(
            **self.config["diffusion_module"]
        )

        self.distogram_head = DistogramHead(
            **self.config["distogram_head"]
        )

        self.confidence_head = ConfidenceHead(
            **self.config["confidence_head"]
        )

        # Projections during recycling
        self.c_token = self.config.input_embedder.c_token
        self.c_z = self.config.input_embedder.c_trunk_pair
        self.recycling_s_proj = nn.Sequential(
            LayerNorm(self.c_token),
            LinearNoBias(self.c_token, self.c_token)
        )
        self.recycling_z_proj = nn.Sequential(
            LayerNorm(self.c_z),
            LinearNoBias(self.c_z, self.c_z)
        )

    def run_trunk(
            self,
            feats: Dict[str, Tensor],
            s_inputs: Tensor,
            s_init: Tensor,
            z_init: Tensor,
            s_prev: Tensor,
            z_prev: Tensor,
            inplace_safe: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        Run a single recycling iteration.
        Args:
            feats:
                dictionary containing the AlphaFold3 features and a
                few additional keys such as "msa_mask" and "token_mask"
            s_inputs:
                [*, N_token, C_token] single inputs embedding from InputFeatureEmbedder
            s_init:
                [*, N_token, C_token] initial token representation
            z_init:
                [*, N_token, N_token, C_z] initial pair representation
            s_prev:
                [*, N_token, C_token] previous token representation from recycling.
                If this is the first iteration, it should be zeros.
            z_prev:
                [*, N_token, N_token, C_z] previous pair representation from recycling.
                If this is the first iteration, it should be zeros.
            inplace_safe:
                whether to use inplace ops
        """

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Prep masks
        token_mask = feats["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Embed the input features
        z = add(z_init, self.recycling_z_proj(z_prev), inplace=inplace_safe)
        s = add(s_init, self.recycling_s_proj(s_prev), inplace=inplace_safe)

        del s_prev, z_prev, s_init, z_init

        # Embed the templates
        z = add(z,
                self.template_embedder(
                    feats,
                    z,
                    pair_mask=pair_mask,
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    inplace_safe=inplace_safe
                ),
                inplace=inplace_safe
                )

        # Process the MSA
        z = add(z,
                self.msa_module(
                    feats=feats,
                    z=z,
                    s_inputs=s_inputs,
                    z_mask=pair_mask,
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    inplace_safe=inplace_safe
                ),
                inplace=inplace_safe
                )

        # Run the pairformer stack
        s, z = self.pairformer(
            s, z,
            single_mask=token_mask,
            pair_mask=pair_mask,
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe
        )
        return s, z

    def _disable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.paiformer.blocks_per_ckpt = None
        self.msa_module.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.pairformer.blocks_per_ckpt = (
            self.config.pairformer_stack.blocks_per_ckpt
        )

        self.msa_module.blocks_per_ckpt = (
            self.config.msa_module.blocks_per_ckpt
        )

    def run_confidence_head(
            self,
            batch: Dict[str, Tensor],
            atom_positions: Tensor,  # [bs, N_atoms, 3]
            s_inputs: Tensor,  # [bs, N_token, C_token]
            s_trunk: Tensor,  # [bs, N_token, C_token]
            z_trunk: Tensor,  # [bs, N_token, N_token, C_z]
            inplace_safe: bool = False
    ):
        """Runs the confidence head on the trunk outputs and sampled atom positions."""
        batch_size, n_tokens, _ = s_trunk.shape

        # Stop gradients
        atom_positions = atom_positions.detach()
        s_inputs = s_inputs.detach()
        s_trunk = s_trunk.detach()
        z_trunk = z_trunk.detach()

        # Gather representative atoms for the confidence head
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_tokens)
        token_repr_atom_indices = batch["token_repr_atom"]
        representative_atoms = atom_positions[batch_indices, token_repr_atom_indices]

        # Compute masks for the confidence head
        single_mask = batch["token_mask"]  # [bs, N_token]
        pair_mask = single_mask[..., None] * single_mask[..., None, :]  # [bs, N_token, N_token]

        # Run the confidence head
        confidences = self.confidence_head.forward(
            s_inputs=s_inputs,
            s=s_trunk,
            z=z_trunk,
            x_repr=representative_atoms,
            single_mask=single_mask,
            pair_mask=pair_mask,
            chunk_size=self.globals.chunk_size,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe
        )
        return confidences

    def forward(self, batch, training: bool = True) -> Dict[str, Tensor]:
        """
        Args:
            batch:
                Dictionary of arguments in AlphaFold3 as outlined in the supplementary
                information. Keys must include the official names of the features in the
                supplement subsection 2.8, Table 5.

                Certain MSA features have their final dimension N_cycle, since they have
                a different set of features for every recycling iteration.

                # Token-wise features
                "residue_index" ([*, N_token]):
                    Residue number in the token’s original input chain.
                "token_index" ([*, N_token]):
                    Token number. Increases monotonically; does not restart at 1 for
                    new chains.
                "asym_id" ([*, N_token]):
                    Unique integer for each distinct chain.
                "entity_id" ([*, N_token]):
                    Unique integer for each distinct sequence.
                "sym_id" ([*, N_token]):
                    Unique integer within chains of this sequence. E.g. if chains A, B
                    and C share a sequence but D does not, their sym_ids would be [0, 1, 2, 0].
                "restype" ([*, N_token, 32]):
                    One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
                    4 RNA nucleotides + unknown, 4 DNA nucleotides + un- known, and gap. Ligands
                    represented as “unknown amino acid”.
                "is_protein" ([*, N_token]):
                    mask indicating is protein
                "is_rna" ([*, N_token]):
                    mask indicating is RNA
                "is_dna" ([*, N_token]):
                    mask indicating is DNA
                "is_ligand" ([*, N_token]):
                    mask indicating is ligand
                "token_mask" ([*, N_token]):
                    binary mask indicating valid (non-padding) tokens

                # Atom-wise features
                "ref_pos" ([*, N_atom]):
                    Atom positions in the reference conformer, with a random rotation and translation
                    applied. Atom positions are given in Å.
                "ref_mask" ([*, N_atom]):
                    Mask indicating which atom slots are used in the reference conformer.
                "ref_element" ([*, N_atom, 128]):
                    One-hot encoding of the element atomic number for each atom in the reference
                    conformer, up to atomic number 128.
                "ref_charge" ([*, N_atom]):
                    Charge for each atom in the reference conformer.
                "ref_atom_name_chars" ([*, N_atom, 4, 64]):
                    One-hot encoding of the unique atom names in the reference conformer.
                    Each character is encoded as ord(c) − 32, and names are padded to length 4.
                "ref_space_uid" ([*, N_atom]):
                    Numerical encoding of the chain id and residue index associated with this
                    reference conformer. Each (chain id, residue index) tuple is as- signed an
                    integer on first appearance.
                "atom_mask" ([*, N_atom]):
                    Binary mask indicating which atoms are valid (non-padding).
                "atom_exists" ([*, N_atom]):
                    Binary mask indicating which atoms exist in the ground truth structure, used for
                    loss masking. It is different from the atom_mask because an atom can be valid but
                    not exist in the crystal structure.

                # MSA features
                "msa" ([*, N_msa, N_token, 32, N_cycle]):
                    One-hot encoding of the processed MSA, using the same classes as restype.
                "has_deletion" ([*, N_msa, N_token, N_cycle]):
                    Binary feature indicating if there is a deletion to the left of each position
                    in the MSA.
                "deletion_value" ([*, N_msa, N_token, N_cycle]):
                    Raw deletion counts (the number of deletions to the left of each MSA position)
                    are transformed to [0, 1] using 2/π * arctan(d/3).
                "msa_mask" ([*, N_msa, N_token, N_cycle]):
                    Binary mask indicating which positions in the MSA are valid.

                "profile" ([*, N_token, 32]):
                    Distribution across restypes in the main MSA. Computed before MSA processing.
                "deletion_mean" ([*, N_token]):
                    Mean number of deletions at each position in the main MSA. Computed before MSA
                    processing


                # Template features
                "template_restype" ([*, N_templ, N_token]):
                    One-hot encoding of the template sequence, see restype.
                "template_pseudo_beta_mask" ([*, N_templ, N_token]):
                    Mask indicating if the Cβ (Cα for glycine) has coordinates for the template
                    at this residue.
                "template_backbone_frame_mask" ([*, N_templ, N_token]):
                    Mask indicating if coordinates exist for all atoms required to compute the
                    backbone frame (used in the template_unit_vector feature).
                "template_distogram" ([*, N_templ, N_token, N_token, 39]):
                    A one-hot pairwise feature indicating the distance between Cß atoms. Pairwise
                    distances are discretized into 38 bins of equal width between 3.25 Å and 50.75 Å;
                    one more bin contains any larger distances.
                "template_unit_vector" ([*, N_templ, N_token, N_token, 3]):
                    The unit vector of the displacement of the Cα atom of all residues within the
                    local frame of each residue. Local frames are computed as in [1].

                # Bonds
                "token_bonds" ([*, N_token, N_token]):
                    A 2D matrix indicating if there is a bond between any atom in token i and token j,
                    restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å
                    during training.

                # Mapping features
                "atom_to_token" ([*, N_atoms]):
                    Token index for each atom in the flat atom representation.
                "token_atom_idx" ([*, N_atoms]):
                    Maps the flat-atom index l to the within-token atom index {1, ..., N_max_atoms_per_token}
                "token_repr_atom" ([*, N_token]):
                    Index of the representative atom in the flat atom representation for each token.

                # Training time features
                "atom_positions" ([*, N_atoms, 3]):
                    Ground truth atom positions in Å.
            training:
                Whether the model is in training mode.
        """
        # Extract number of recycles
        n_cycle = batch['msa'].shape[-1]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())  # TODO: self.training is None

        # Embed input features: relpos encoding, token_bonds etc.
        s_inputs, s_init, z_init = self.input_embedder(batch, inplace_safe=inplace_safe)

        is_grad_enabled = torch.is_grad_enabled()

        # Initialize recycling embeddings as zeros
        s_prev = s_init.new_zeros(s_init.shape)  # torch.zeros_like(s_init)
        z_prev = z_init.new_zeros(z_init.shape)  # torch.zeros_like(z_init)

        def get_recycling_features(index):
            """Convenience method that extracts the MSA features given the recycling index."""
            recycling_dict = {}
            special_keys = ["msa", "msa_mask", "has_deletion", "deletion_value"]
            for key, tensor in batch.items():
                if key in special_keys:
                    # Get a view of the tensor for the current cycle
                    recycling_dict[key] = tensor[..., index]
                else:
                    # Use the original tensor
                    recycling_dict[key] = tensor

            return recycling_dict

        # Main recycling loop
        for cycle_no in range(n_cycle):
            # Select the features for the current recycling cycle
            feats = get_recycling_features(cycle_no)

            # Enable grad if we're training, and it's the final recycling layer
            is_final_iter = cycle_no == (n_cycle - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the trunk
                s, z = self.run_trunk(
                    feats=feats,
                    s_inputs=s_inputs,
                    s_init=s_init,
                    z_init=z_init,
                    s_prev=s_prev,
                    z_prev=z_prev,
                    inplace_safe=inplace_safe
                )
                if not is_final_iter:
                    s_prev = s
                    z_prev = z

        # Output dictionary
        outputs = {}

        # Add asym_id for alignment of the output
        if "asym_id" in batch:
            outputs["asym_id"] = batch["asym_id"]

        # Run the diffusion module
        n_steps = 200
        rollout_samples_per_trunk = self.globals.samples_per_trunk
        if training:
            n_steps = 20  # Mini roll-out for training
            rollout_samples_per_trunk = 1  # only do a single rollout sample per trunk during training

            # Run the diffusion module once for denoising during training
            denoised_atoms = self.diffusion_module.training(
                ground_truth_atoms=batch["atom_positions"],
                features=batch,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                samples_per_trunk=self.globals.samples_per_trunk,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention
            )
            # Add the denoised atoms to the output dictionary
            outputs["denoised_atoms"] = denoised_atoms

        # Diffusion roll-out
        sampled_positions = self.diffusion_module.sample(
            features=batch,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            n_steps=n_steps,
            samples_per_trunk=rollout_samples_per_trunk,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention
        )
        outputs["sampled_positions"] = sampled_positions

        # Run heads
        outputs["logits_distogram"] = self.distogram_head.forward(z)

        # Run confidence head with stop-gradient
        confidences = self.run_confidence_head(
            batch=batch,
            atom_positions=sampled_positions,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            inplace_safe=inplace_safe
        )
        # update the outputs dictionary with the confidence head outputs
        outputs.update(confidences)
        return outputs
