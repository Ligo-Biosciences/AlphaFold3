"""AlphaFold3 model implementation."""
import torch
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm
from src.models.embedders import InputEmbedder, TemplateEmbedder
from src.models.pairformer import PairformerStack
from src.models.msa_module import MSAModule
from src.models.diffusion_module import DiffusionModule
from src.models.heads import DistogramHead, ConfidenceHead
from src.models.components.primitives import LinearNoBias
from src.utils.tensor_utils import add, tensor_tree_map
from typing import Tuple, Dict


class AlphaFold3(nn.Module):
    def __init__(self, config):
        super(AlphaFold3, self).__init__()
        self.globals = config.globals
        self.config = config.model

        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"]
        )

        # self.template_embedder = TemplateEmbedder(
        #    **self.config["template_embedder"]
        # )

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

        # self.confidence_head = ConfidenceHead(
        #    **self.config["confidence_head"]
        # )

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

    # @torch.compile  # (mode="max-autotune")
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
        # z = add(z,
        #        self.template_embedder(
        #            feats,
        #            z,
        #            pair_mask=pair_mask,
        #            chunk_size=self.globals.chunk_size,
        #            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
        #            inplace_safe=inplace_safe
        #        ),
        #        inplace=inplace_safe
        #        )

        # Process the MSA
        z = add(z,
                self.msa_module(
                    feats=feats,
                    z=z,
                    s_inputs=s_inputs,
                    z_mask=pair_mask,
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
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
            inplace_safe=inplace_safe
        )
        return s, z

    def _disable_activation_checkpointing(self):
        # self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.pairformer.blocks_per_ckpt = None
        self.msa_module.blocks_per_ckpt = None
        self.diffusion_module.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        # self.template_embedder.template_pair_stack.blocks_per_ckpt = (
        #    self.config.template.template_pair_stack.blocks_per_ckpt
        # )
        self.pairformer.blocks_per_ckpt = (
            self.config.pairformer_stack.blocks_per_ckpt
        )
        self.msa_module.blocks_per_ckpt = (
            self.config.msa_module.blocks_per_ckpt
        )
        self.diffusion_module.blocks_per_ckpt = (
            self.config.diffusion_module.blocks_per_ckpt
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
        batch_indices = torch.arange(batch_size).reshape(batch_size, 1)
        token_repr_atom_indices = batch["token_repr_atom"]
        representative_atoms = atom_positions[batch_indices, token_repr_atom_indices, :]

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

                The final dimension of each input must have length equal to
                the number of recycling iterations.
                Features (without the recycling dimension):

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
                "aatype" ([*, N_token]):
                    Encoding of the sequence. 21 possible values: 20 amino acids + unknown.
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
                "ref_pos" ([*, N_atom, 3]):
                    Atom positions in the reference conformer, with a random rotation and translation
                    applied. Atom positions are given in Å.
                "ref_mask" ([*, N_atom]):
                    Mask indicating which atom slots are used in the reference conformer.
                "ref_element" ([*, N_atom, 4]):
                    One-hot encoding of the element atomic number for each atom in the reference
                    conformer, limited to the 4 elements in proteins for now.
                "ref_charge" ([*, N_atom]):
                    Charge for each atom in the reference conformer.
                "ref_atom_name_chars" ([*, N_atom, 4, 64]):
                    One-hot encoding of the unique atom names in the reference conformer.
                    Each character is encoded as ord(c) − 32, and names are padded to length 4.
                "ref_space_uid" ([*, N_atom]):
                    Numerical encoding of the chain id and residue index associated with this
                    reference conformer. Each (chain id, residue index) tuple is assigned an
                    integer on first appearance.

                # MSA features
                "msa_feat" ([*, N_msa, N_token, 49]):
                    Concatenated MSA features that are the same as in AlphaFold2.
                "msa_mask" ([*, N_msa, N_token]):
                    Binary mask indicating which positions in the MSA are valid.

                # Mapping features
                "atom_to_token" ([*, N_atoms]):
                    Token index for each atom in the flat atom representation.
                "token_atom_idx" ([*, N_atoms]):
                    Maps the flat-atom index l to the within-token atom index {1, ..., N_max_atoms_per_token}
                "token_repr_atom" ([*, N_token]):
                    Index of the representative atom in the flat atom representation for each token.

                # Training time features
                "all_atom_positions" ([*, N_atoms, 3]):
                    Ground truth atom positions in Å.
                "atom_mask" ([*, N_atoms]):
                    Mask indicating which atom slots are used in the ground truth structure.
                "atom_exists" ([*, N_atoms, 3]):
                    Mask indicating which atom slots exist in the ground truth structure.
            training:
                Whether the model is in training mode.
        """

        # Extract number of recycles
        n_cycle = batch['msa_feat'].shape[-1]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (training or torch.is_grad_enabled())

        # Extract features without the recycling dimension
        feats = tensor_tree_map(lambda t: t[..., -1], batch)

        # Embed input features: relpos encoding, token_bonds etc.
        s_inputs, s_init, z_init = self.input_embedder(
            feats,
            inplace_safe=inplace_safe,
        )

        is_grad_enabled = torch.is_grad_enabled()

        # Initialize recycling embeddings as zeros
        s_prev = s_init.new_zeros(s_init.shape)  # torch.zeros_like(s_init)
        z_prev = z_init.new_zeros(z_init.shape)  # torch.zeros_like(z_init)

        # Main recycling loop
        for cycle_no in range(n_cycle):
            # Select the features for the current recycling cycle
            feats = tensor_tree_map(lambda t: t[..., cycle_no], batch)  # Remove recycling dimension

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

        # Remove the recycling dimension from the batch
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Output dictionary
        outputs = {}

        # Add asym_id for alignment of the output
        if "asym_id" in batch:
            outputs["asym_id"] = batch["asym_id"]

        # Run the diffusion module
        n_steps = 200
        if training:
            n_steps = 20  # Mini roll-out for training

            # Run the diffusion module once for denoising during training
            diff_output = self.diffusion_module.train_step(
                ground_truth_atoms=batch["all_atom_positions"],
                features=batch,
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                samples_per_trunk=self.globals.samples_per_trunk,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            )
            # Add the denoised atoms, timesteps, and augmented gt atoms for loss calculation
            outputs.update(diff_output)

        # Diffusion roll-out without gradients
        sampled_positions = self.diffusion_module.sample(
            features=batch,
            s_inputs=s_inputs.detach(),
            s_trunk=s.detach(),
            z_trunk=z.detach(),
            n_steps=n_steps,
            samples_per_trunk=1,  # only a single sample during rollout
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention
        )
        outputs["sampled_positions"] = sampled_positions

        # Run heads
        outputs["distogram_logits"] = self.distogram_head.forward(z)

        # Run confidence head with stop-gradient  # TODO: there is a bug in the confidence head
        # confidences = self.run_confidence_head(
        #    batch=batch,
        #    atom_positions=sampled_positions,
        #    s_inputs=s_inputs,
        #    s_trunk=s,
        #    z_trunk=z,
        #    inplace_safe=inplace_safe
        # )
        # update the outputs dictionary with the confidence head outputs
        # outputs.update(confidences)
        return outputs
