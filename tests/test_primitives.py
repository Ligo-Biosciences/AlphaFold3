import unittest
import torch
from torch import nn
from src.models.components.primitives import AdaLN, Linear, LinearNoBias, safe_softmax, _attention, _deepspeed_evo_attn


class TestAdaLN(unittest.TestCase):
    def setUp(self):
        self.model = AdaLN(normalized_shape=10)

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

    def test_forward_pass_different_shapes(self):
        a = torch.randn(3, 10)
        s = torch.randn(3, 10)
        output = self.model(a, s)
        self.assertEqual(output.shape, a.shape)

    def test_forward_pass_single_element(self):
        a = torch.randn(1, 10)
        s = torch.randn(1, 10)
        output = self.model(a, s)
        self.assertEqual(output.shape, a.shape)

    def test_forward_pass_large_input(self):
        a = torch.randn(1000, 10)
        s = torch.randn(1000, 10)
        output = self.model(a, s)
        self.assertEqual(output.shape, a.shape)

    def test_forward_pass_zero_input(self):
        a = torch.zeros(2, 10)
        s = torch.zeros(2, 10)
        output = self.model(a, s)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_forward_pass_extreme_values(self):
        a = torch.full((2, 10), 1e6)
        s = torch.full((2, 10), -1e6)
        output = self.model(a, s)
        self.assertEqual(output.shape, a.shape)


class TestLinear(unittest.TestCase):
    def setUp(self):
        self.model = Linear(10, 5)

    def test_forward_pass(self):
        x = torch.randn(2, 10)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_forward_pass_zero_input(self):
        x = torch.zeros(2, 10)
        output = self.model(x)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_forward_pass_extreme_values(self):
        x = torch.full((2, 10), 1e6)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 5))


class TestLinearNoBias(unittest.TestCase):
    def setUp(self):
        self.model = LinearNoBias(10, 5)

    def test_forward_pass(self):
        x = torch.randn(2, 10)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 5))

    def test_forward_pass_zero_input(self):
        x = torch.zeros(2, 10)
        output = self.model(x)
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_forward_pass_extreme_values(self):
        x = torch.full((2, 10), 1e6)
        output = self.model(x)
        self.assertEqual(output.shape, (2, 5))


class TestSafeSoftmax(unittest.TestCase):
    def test_safe_softmax(self):
        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        output = safe_softmax(x, axis=-1)
        self.assertTrue(torch.allclose(output, torch.tensor([[0.5, 0.5], [0.5, 0.5]])))

    def test_safe_softmax_zero_input(self):
        x = torch.zeros(2, 2)
        output = safe_softmax(x, axis=-1)
        self.assertTrue(torch.allclose(output, torch.tensor([[0.5, 0.5], [0.5, 0.5]])))


class TestAttention(unittest.TestCase):
    def test_attention(self):
        query = torch.randn(2, 2, 2, 3)
        key = torch.randn(2, 2, 2, 3)
        value = torch.randn(2, 2, 2, 3)
        biases = [torch.randn(2, 2, 2, 2)]
        output = _attention(query, key, value, biases)
        self.assertEqual(output.shape, (2, 2, 2, 3))


class TestDeepSpeedEvoAttn(unittest.TestCase):
    def test_deepspeed_evo_attn(self):
        query = torch.randn(2, 2, 2, 3)
        key = torch.randn(2, 2, 2, 3)
        value = torch.randn(2, 2, 2, 3)
        biases = [torch.randn(2, 2, 2, 2)]
        with self.assertRaises(ValueError):
            _deepspeed_evo_attn(query, key, value, biases)


if __name__ == "__main__":
    unittest.main()
