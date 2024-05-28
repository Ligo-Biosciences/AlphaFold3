import unittest
import torch
import torch.nn as nn
from src.models.components.atom_attention import AtomAttentionPairBias


class TestAttentionPairBias(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 128
        self.num_heads = 8
        self.batch_size = 2
        self.n_atoms = 384
        self.c_pair = 16

        # Example inputs
        self.atom_single_repr = torch.rand(self.batch_size, self.n_atoms, self.embed_dim)
        self.atom_single_proj = torch.rand(self.batch_size, self.n_atoms, self.embed_dim)
        self.atom_pair_repr = torch.rand(self.batch_size, self.n_atoms, self.n_atoms, self.c_pair)

    def test_module_instantiation(self):
        """Test instantiation of the module with default parameters."""
        module = AtomAttentionPairBias(embed_dim=self.embed_dim)
        self.assertIsInstance(module, nn.Module)

    def test_forward_output_shape(self):
        """Test the forward function output shape."""
        module = AtomAttentionPairBias(embed_dim=self.embed_dim, num_heads=self.num_heads)
        output = module(self.atom_single_repr, self.atom_single_proj, self.atom_pair_repr)
        expected_shape = (self.batch_size, self.n_atoms, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)


# Run the tests
if __name__ == '__main__':
    unittest.main()
