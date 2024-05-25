"""Construct an initial 1D embedding."""
import torch


class InputFeatureEmbedder(torch.nn.Module):
    """A class that performs attention over all atoms in order to encode the information
    about the chemical structure of all the molecules, leading to a single representation
    representing all the tokens.
    - Embed per-atom features
    - Concatenate the per-token features
    """
    pass
