import unittest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.components.transition import ConditionedTransitionBlock, Transition
from src.models.components.primitives import AdaLN


class TestTransition(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.n = 4
        self.model = Transition(self.input_dim, self.n)
        self.input_tensor = torch.randn(10, self.input_dim)  # Batch of 10

    def test_output_shape(self):
        """Test if the output shape is correct"""
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_layer_norm_applied(self):
        """Test if layer normalization is applied"""
        normed_input = self.model.layer_norm(self.input_tensor)
        self.assertTrue(torch.allclose(normed_input.mean(dim=-1), torch.zeros(normed_input.shape[0]), atol=1e-6),
                        "Mean after LayerNorm should be close to 0")
        self.assertTrue(
            torch.allclose(normed_input.std(dim=-1, unbiased=False), torch.ones(normed_input.shape[0]), atol=1e-6),
            "Std after LayerNorm should be close to 1")

    def test_forward_pass(self):
        """Test if forward pass works without errors"""
        try:
            self.model(self.input_tensor)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

    def test_linear_layers(self):
        """Test if linear layers have correct output dimensions"""
        linear_1_output = self.model.linear_1(self.input_tensor)
        self.assertEqual(linear_1_output.shape, (self.input_tensor.shape[0], self.input_dim * self.n))

        linear_2_output = self.model.linear_2(self.input_tensor)
        self.assertEqual(linear_2_output.shape, (self.input_tensor.shape[0], self.input_dim * self.n))

        combined_output = F.silu(linear_1_output) * linear_2_output
        self.assertEqual(combined_output.shape, (self.input_tensor.shape[0], self.input_dim * self.n))

        final_output = self.model.output_linear(combined_output)
        self.assertEqual(final_output.shape, (self.input_tensor.shape[0], self.input_dim))


class TestConditionedTransitionBlock(unittest.TestCase):
    def setUp(self):
        self.model = ConditionedTransitionBlock(input_dim=10, n=2)

    def test_initialization(self):
        self.assertIsInstance(self.model.ada_ln, AdaLN)
        self.assertIsInstance(self.model.hidden_gating_linear, nn.Linear)
        self.assertIsInstance(self.model.hidden_linear, nn.Linear)
        self.assertIsInstance(self.model.output_linear, nn.Linear)
        self.assertIsInstance(self.model.output_gating_linear, nn.Linear)

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


if __name__ == '__main__':
    unittest.main()
