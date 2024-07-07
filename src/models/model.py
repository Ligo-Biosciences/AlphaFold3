"""AlphaFold3 model implementation."""


class AlphaFold3:
    def __init__(self, config):
        # InputFeatureEmbedder
        # RelativePositionEncoder
        # TemplateEmbedder
        # MsaModule
        # PairformerStack
        # DiffusionModule
        # ConfidenceHead
        pass

    def embed_templates(self, batch, feats, z, pair_mask, template_dim, inplace_safe):
        pass

    def iteration(self, features, prevs, _recycle=True):
        # initialize output dictionary

        # dtype cast the features

        # Grab some data about the input

        # Controls whether the model uses in-place operations throughout
        # Embed the input features

        # Unpack the recycling embeddings. Removing them from the list allows
        # them to be freed further down in this function, saving memory

        # Initialize the recycling embeddings, if needs be

        # Embed the templates

        # Process the MSA

        # Run the pairformer stack

        # outputs, s_prev, z_prev

        pass

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
