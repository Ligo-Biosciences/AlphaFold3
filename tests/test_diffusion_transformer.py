import torch
import unittest
from src.models.diffusion_transformer import DiffusionTransformer


class TestDiffusionTransformer(unittest.TestCase):

    def setUp(self):
        # Setting up a common test environment
        self.c_token = 128
        self.num_heads = 8
        self.batch_size = 2
        self.n_seq = 2
        self.n_tokens = 64
        self.num_blocks = 4
        self.c_pair = 16

        # Initialize the module
        self.module = DiffusionTransformer(
            c_token=self.c_token,
            c_pair=self.c_pair,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout=0.0,
            blocks_per_ckpt=1,
        )

        # Example input tensors
        self.single_repr = torch.randn(self.batch_size, self.n_seq, self.n_tokens, self.c_token)
        self.single_proj = torch.randn(self.batch_size, self.n_tokens, self.c_token)
        self.pair_repr = torch.randn(self.batch_size, self.n_tokens, self.n_tokens, self.c_pair)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))

    def test_module_output_shape(self):
        """Test output shapes from the forward pass."""
        output = self.module(self.single_repr, self.single_proj, self.pair_repr, self.mask,
                             use_deepspeed_evo_attention=False)
        expected_shape = (self.batch_size, self.n_seq, self.n_tokens, self.c_token)
        self.assertEqual(output.shape, expected_shape, "Output shape should match expected shape.")
