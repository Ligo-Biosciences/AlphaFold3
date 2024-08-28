import unittest
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from src.models.components.atom_attention_naive import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    AtomTransformerBlock
)


class TestAtomAttentionEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_atoms = 1536
        self.n_seq = 2

        # Model parameters
        self.n_tokens = 384
        self.c_token = 64
        self.c_atom = 128
        self.c_atompair = 16
        self.c_trunk_pair = 16
        self.no_blocks = 3
        self.no_heads = 4
        self.dropout = 0.1
        self.n_queries = 32
        self.n_keys = 128
        self.trunk_conditioning = True
        self.encoder = AtomAttentionEncoder(
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_trunk_pair=self.c_trunk_pair,
            no_blocks=self.no_blocks,
            no_heads=self.no_heads,
            dropout=self.dropout,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            trunk_conditioning=self.trunk_conditioning
        )

    def test_initialization(self):
        """Test whether the module initializes with the correct properties."""
        self.assertEqual(self.encoder.c_atom, self.c_atom)
        self.assertTrue(isinstance(self.encoder.linear_atom_embedding, nn.Linear))
        # Add more assertions for other properties

    def test_forward_dimensions(self):
        """Test the forward pass with mock data to ensure output dimensions."""
        features = {
            'ref_pos': torch.rand(self.batch_size, self.n_atoms, 3),
            'ref_charge': torch.rand(self.batch_size, self.n_atoms),
            'ref_mask': torch.ones(self.batch_size, self.n_atoms),
            'ref_element': torch.rand(self.batch_size, self.n_atoms, 4),
            'ref_atom_name_chars': torch.randint(0, 2, (self.batch_size, self.n_atoms, 4)),
            'ref_space_uid': torch.randint(0, self.n_atoms, (self.batch_size, self.n_atoms)),
            'atom_to_token': torch.randint(0, self.n_tokens, (self.batch_size, self.n_atoms)),
        }
        noisy_pos = torch.rand(self.batch_size, self.n_seq, self.n_atoms, 3)

        # Pairformer outputs (adjust as per actual module expectations)
        s_trunk = torch.rand(self.batch_size, self.n_tokens, self.c_token)
        z_trunk = torch.rand(self.batch_size, self.n_tokens, self.n_tokens, self.c_trunk_pair)
        mask = torch.randint(0, 2, (self.batch_size, self.n_atoms))

        output = self.encoder(
            features=features,
            n_tokens=self.n_tokens,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noisy_pos=noisy_pos,
            mask=mask,
        )
        self.assertEqual(output.token_single.shape, (self.batch_size, self.n_seq, self.n_tokens, self.c_token))
        self.assertEqual(output.atom_single_skip_repr.shape, (self.batch_size, self.n_seq, self.n_atoms, self.c_atom))
        self.assertEqual(output.atom_single_skip_proj.shape, (self.batch_size, self.n_atoms, self.c_atom))
        self.assertEqual(output.atom_pair_skip_repr.shape, (self.batch_size,
                                                            self.n_atoms,
                                                            self.n_atoms,
                                                            self.c_atompair))


class TestAtomAttentionDecoder(unittest.TestCase):
    def setUp(self):
        self.c_atompair = 16
        self.c_atom = 128
        self.c_token = 64
        self.decoder = AtomAttentionDecoder(
            c_token=self.c_token,
            no_blocks=2,
            no_heads=4,
            dropout=0.1,
            n_queries=32,
            n_keys=128,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
        )
        self.bs = 3  # Batch size
        self.n_tokens = 384
        self.n_atoms = 1024
        self.n_queries = 32
        self.n_keys = 128
        self.n_seq = 2

    def test_forward_dimensions(self):
        """Test the output dimensions from the forward pass."""
        token_repr = torch.randn(self.bs, self.n_seq, self.n_tokens, self.decoder.c_token)
        atom_single_skip_repr = torch.randn(self.bs, self.n_seq, self.n_atoms, self.decoder.c_atom)
        atom_single_skip_proj = torch.randn(self.bs, self.n_atoms, self.decoder.c_atom)
        atom_pair_skip_repr = torch.randn(
            (self.bs, self.n_atoms, self.n_atoms, self.c_atompair)
        )
        tok_idx = torch.randint(0, self.n_tokens, (self.bs, self.n_atoms))
        mask = torch.randint(0, 2, (self.bs, self.n_atoms))

        output = self.decoder(
            token_repr,
            atom_single_skip_repr,
            atom_single_skip_proj,
            atom_pair_skip_repr,
            tok_idx,
            mask,
        )

        self.assertEqual(output.shape, (self.bs, self.n_seq, self.n_atoms, 3))


def test_prep_betas_visualization():
    # Set up parameters
    n_atoms = 1024
    device = torch.device("cpu")  # Use CPU for this test

    # Create an instance of AtomTransformerBlock
    block = AtomTransformerBlock(c_atom=64)  # Use default values for other parameters

    # Get the beta matrix
    beta = block._prep_betas(n_atoms, device)

    # Convert to numpy for visualization
    beta_np = beta.numpy()

    # Create a mask for 0 and -inf values
    mask_zero = (beta_np == 0)
    mask_inf = np.isinf(beta_np)

    # Create a custom colormap
    cmap = plt.cm.colors.ListedColormap(['red', 'blue'])
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mask_zero.astype(int), cmap=cmap, norm=norm, interpolation='nearest')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['-inf', '0'])

    # Set title and labels
    ax.set_title(f"Beta Matrix Visualization (N_tokens={n_atoms})")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    # Save the plot
    plt.savefig("beta_matrix_visualization.png")
    plt.close()

    # Print some statistics
    print(f"Total elements: {n_atoms * n_atoms}")
    print(f"Number of 0s: {mask_zero.sum()}")
    print(f"Number of -inf: {mask_inf.sum()}")

    # assert mask_zero.sum() + mask_inf.sum() == n_tokens * n_tokens, "All elements should be either 0 or -inf"

if __name__ == "__main__":
    test_prep_betas_visualization()
    unittest.main()