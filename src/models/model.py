"""AlphaFold3 model implementation."""
import torch
from torch import nn
from torch import Tensor
from src.models.embedders import InputEmbedder, TemplateEmbedder
from src.models.pairformer import PairformerStack
from src.models.msa_module import MSAModule
from src.models.diffusion_module import DiffusionModule
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

        self.pairformer_stack = PairformerStack(
            **self.config["pairformer_stack"]
        )

        self.diffusion_module = DiffusionModule(
            **self.config["diffusion_module"]
        )

        # Projections during recycling
        c_token = self.config.input_embedder.c_token
        c_trunk_pair = self.config.input_embedder.c_trunk_pair
        self.recycling_s_proj = nn.Sequential(
            LayerNorm(c_token),
            LinearNoBias(c_token, c_token)
        )
        self.recycling_z_proj = nn.Sequential(
            LayerNorm(c_trunk_pair),
            LinearNoBias(c_trunk_pair, c_trunk_pair)
        )

    def run_trunk(
            self,
            feats: Dict[str, Tensor],
            s_inputs: Tensor,
            s_init: Tensor,
            z_init: Tensor,
            s_prev: Tensor,
            z_prev: Tensor
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
        """

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

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
        pass

    def _enable_activation_checkpointing(self):
        pass

    def forward(self, inputs):
        # Initialize recycling embeddings

        # embed input features: relpos encoding, token_bonds etc.

        # Main recycling loop
        # for cycle_no in range(num_iters):
        #    select features for this iteration
        #    # Enable grad if we're training and it's the final recycling layer

        pass
