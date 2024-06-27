import unittest
import torch
import torch.nn as nn
from src.models.components.atom_attention import (
    AtomAttentionPairBias, AtomAttentionEncoder, AtomAttentionDecoder
)


class TestAttentionPairBias(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 128
        self.num_heads = 8
        self.batch_size = 2
        self.n_atoms = 384
        self.c_pair = 16
        self.n_queries = 32
        self.n_keys = 128

        # Example inputs
        self.atom_single_repr = torch.randn(self.batch_size, self.n_atoms, self.embed_dim)
        self.atom_single_proj = torch.randn(self.batch_size, self.n_atoms, self.embed_dim)
        self.atom_pair_local = torch.randn(self.batch_size,
                                           self.n_atoms // self.n_queries,
                                           self.n_queries,
                                           self.n_keys,
                                           self.c_pair)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_atoms))

    def test_module_instantiation(self):
        """Test instantiation of the module with default parameters."""
        module = AtomAttentionPairBias(c_atom=self.embed_dim)
        self.assertIsInstance(module, nn.Module)

    def test_forward_output_shape(self):
        """Test the forward function output shape."""
        module = AtomAttentionPairBias(c_atom=self.embed_dim, num_heads=self.num_heads)
        output = module(self.atom_single_repr, self.atom_single_proj, self.atom_pair_local, self.mask)
        expected_shape = (self.batch_size, self.n_atoms, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)

    def test_mask(self):
        # TODO: implement a test for masking
        self.assertTrue(True)


class TestAtomAttentionEncoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_atoms = 1536

        # Model parameters
        self.n_tokens = 384
        self.c_token = 64
        self.c_atom = 128
        self.c_atompair = 16
        self.c_trunk_pair = 16
        self.num_blocks = 3
        self.num_heads = 4
        self.dropout = 0.1
        self.n_queries = 32
        self.n_keys = 128
        self.trunk_conditioning = True
        self.encoder = AtomAttentionEncoder(
            n_tokens=self.n_tokens,
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_trunk_pair=self.c_trunk_pair,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=self.dropout,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            trunk_conditioning=self.trunk_conditioning
        )

    def test_initialization(self):
        """Test whether the module initializes with the correct properties."""
        self.assertEqual(self.encoder.n_tokens, self.n_tokens)
        self.assertEqual(self.encoder.c_atom, self.c_atom)
        self.assertTrue(isinstance(self.encoder.linear_atom_embedding, nn.Linear))
        # Add more assertions for other properties

    def test_forward_dimensions(self):
        """Test the forward pass with mock data to ensure output dimensions."""
        features = {
            'ref_pos': torch.rand(self.batch_size, self.n_atoms, 3),
            'ref_charge': torch.rand(self.batch_size, self.n_atoms),
            'ref_mask': torch.ones(self.batch_size, self.n_atoms),
            'ref_element': torch.rand(self.batch_size, self.n_atoms, 128),
            'ref_atom_name_chars': torch.randint(0, 2, (self.batch_size, self.n_atoms, 4, 64)),
            'ref_space_uid': torch.randint(0, self.n_atoms, (self.batch_size, self.n_atoms)),
            'atom_to_token': torch.randint(0, self.n_tokens, (self.batch_size, self.n_atoms)),
        }
        noisy_pos = torch.rand(self.batch_size, self.n_atoms, 3)

        # Pairformer outputs (adjust as per actual module expectations)
        s_trunk = torch.rand(self.batch_size, self.n_tokens, self.c_token)
        z_trunk = torch.rand(self.batch_size, self.n_tokens, self.n_tokens, self.c_trunk_pair)
        mask = torch.randint(0, 2, (self.batch_size, self.n_atoms))
        # pairformer_output = {
        #     'token_single': torch.rand(self.batch_size, self.n_tokens, self.c_token),
        #     'token_pair': torch.rand(self.batch_size, self.n_tokens, self.n_tokens, self.c_trunk_pair)
        # }

        output = self.encoder(features, s_trunk, z_trunk, noisy_pos, mask)
        self.assertEqual(output.token_single.shape, torch.Size([self.batch_size, self.n_tokens, self.c_token]))
        self.assertEqual(output.atom_single_skip_repr.shape, torch.Size([self.batch_size, self.n_atoms, self.c_atom]))
        self.assertEqual(output.atom_single_skip_proj.shape, torch.Size([self.batch_size, self.n_atoms, self.c_atom]))
        self.assertEqual(output.atom_pair_skip_repr.shape, torch.Size([self.batch_size,
                                                                       self.n_atoms // self.n_queries,
                                                                       self.n_queries,
                                                                       self.n_keys,
                                                                       self.c_atompair]))


class TestAtomAttentionDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = AtomAttentionDecoder(
            c_token=64,
            num_blocks=2,
            num_heads=4,
            dropout=0.1,
            n_queries=32,
            n_keys=128,
            c_atom=128,
            c_atompair=16
        )
        self.bs = 3  # Batch size
        self.n_tokens = 384
        self.n_atoms = 1024
        self.n_queries = 32
        self.n_keys = 128

    def test_initialization(self):
        """Test the correct initialization of AtomAttentionDecoder."""
        self.assertEqual(self.decoder.num_blocks, 2)
        self.assertEqual(self.decoder.num_heads, 4)
        self.assertEqual(self.decoder.dropout, 0.1)
        self.assertEqual(self.decoder.n_queries, 32)
        self.assertEqual(self.decoder.n_keys, 128)
        self.assertEqual(self.decoder.c_atom, 128)
        self.assertEqual(self.decoder.c_atompair, 16)

    def test_forward_dimensions(self):
        """Test the output dimensions from the forward pass."""
        token_repr = torch.randn(self.bs, self.n_tokens, self.decoder.c_token)
        atom_single_skip_repr = torch.randn(self.bs, self.n_atoms, self.decoder.c_atom)
        atom_single_skip_proj = torch.randn(self.bs, self.n_atoms, self.decoder.c_atom)
        atom_pair_skip_repr = torch.randn(self.bs,
                                          self.n_atoms // self.n_queries,
                                          self.n_queries,
                                          self.n_keys,
                                          self.decoder.c_atompair)
        tok_idx = torch.randint(0, self.n_tokens, (self.bs, self.n_atoms))
        mask = torch.randint(0, 2, (self.bs, self.n_atoms))

        output = self.decoder(
            token_repr,
            atom_single_skip_repr,
            atom_single_skip_proj,
            atom_pair_skip_repr,
            tok_idx,
            mask
        )

        self.assertEqual(output.shape, (self.bs, self.n_atoms, 3))


# Run the tests
if __name__ == '__main__':
    unittest.main()
