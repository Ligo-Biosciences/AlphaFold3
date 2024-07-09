import unittest
import torch
from torch import nn
from src.models.components.primitives import AdaLN, AttentionPairBias


class TestAdaLN(unittest.TestCase):
    def setUp(self):
        self.model = AdaLN(normalized_shape=10)

    def test_initialization(self):
        self.assertIsInstance(self.model.a_layer_norm, nn.LayerNorm)
        self.assertIsInstance(self.model.s_layer_norm, nn.LayerNorm)
        self.assertIsInstance(self.model.gating_linear, nn.Linear)
        self.assertIsInstance(self.model.skip_linear, nn.Linear)

    def test_forward_pass(self):
        a = torch.randn(2, 10)
        s = torch.randn(2, 10)
        output = self.model(a, s)
        self.assertEqual(output.shape, a.shape)

    def test_gradient_computation(self):
        a = torch.randn(2, 10, requires_grad=True)
        s = torch.randn(2, 10, requires_grad=True)
        output = self.model(a, s)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(s.grad)


class TestAttentionPairBias(unittest.TestCase):

    def setUp(self):
        # Setting up a common test environment
        self.embed_dim = 128
        self.num_heads = 8
        self.batch_size = 2
        self.n_tokens = 64
        self.c_pair = 16

        # Initialize the module
        self.module = AttentionPairBias(dim=self.embed_dim, no_heads=self.num_heads, c_pair=self.c_pair)

        # Example x tensors
        self.single_repr = torch.randn(self.batch_size, self.n_tokens, self.embed_dim)
        self.single_proj = torch.randn(self.batch_size, self.n_tokens, self.embed_dim)
        self.pair_repr = torch.randn(self.batch_size, self.n_tokens, self.n_tokens, self.c_pair)
        self.mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))

    def test_module_output_shape(self):
        """Test output shapes from the forward pass."""
        output = self.module(self.single_repr, self.single_proj, self.pair_repr, self.mask)
        expected_shape = (self.batch_size, self.n_tokens, self.embed_dim)
        self.assertEqual(output.shape, expected_shape, "Output shape should match expected shape.")

    def test_module_initialization(self):
        """Test proper initialization of the module."""
        self.assertEqual(self.module.dim, 128, "Embedding dimension should match initialization.")
        self.assertEqual(self.module.num_heads, 8, "Number of heads should match initialization.")

    def test_parameter_initialization_values(self):
        """Test custom initialization values of parameters."""
        # Check initial value of output projection bias
        output_bias_initial_value = self.module.output_proj_linear.bias.data.mean().item()
        self.assertAlmostEqual(output_bias_initial_value, -2.0, places=5,
                               msg="Output projection bias should be initialized to -2.")


if __name__ == "__main__":
    unittest.main()
