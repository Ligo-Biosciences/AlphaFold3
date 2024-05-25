import unittest
import torch
from torch import nn
from src.models.components.primitives import AdaLN


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


if __name__ == "__main__":
    unittest.main()
