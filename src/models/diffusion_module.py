"""
Diffusion Module from AlphaFold3.
The StructureModule of AlphaFold2 using invariant point attention was replaced with a relatively standard
non-equivariant point-cloud diffusion model over all atoms. The denoiser is based on a modern transformer,
but with several modifications to make it more amenable to the task. The main changes are:
 - Conditioning from the trunk in several ways: we initialise the activations for the single embedding, use
    a variant of Adaptive Layernorm for the single conditioning and logit biasing for the pair conditioning.
 - Standard transformer tricks (e.g. SwiGLU) and methods used in AlphaFold2 (gating)
 - A two-level architecture, working first on atoms, then tokens, then atoms again.
"""
import torch


class DiffusionTransformer(torch.nn.Module):
    pass


class DiffusionConditioning(torch.nn.Module):
    pass
