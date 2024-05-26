import unittest
import torch
from src.diffusion.conditioning import FourierEmbedding


class TestFourierEmbedding(unittest.TestCase):
    def setUp(self):
        self.embed_dim = 128
        self.model = FourierEmbedding(self.embed_dim)

    def test_embedding_shape(self):
        """Test if the output shape is correct"""
        timesteps = torch.tensor([[0], [1], [2], [3], [4], [5]], dtype=torch.float32)
        embeddings = self.model(timesteps)
        self.assertEqual(embeddings.shape, (timesteps.shape[0], self.embed_dim))

    def test_different_timesteps(self):
        """Test if different timesteps produce different embeddings"""
        timesteps = torch.tensor([[0], [1]], dtype=torch.float32)
        embeddings = self.model(timesteps)
        self.assertFalse(torch.allclose(embeddings[0], embeddings[1]), "Embeddings for different timesteps should not "
                                                                       "be identical")

    def test_constant_output(self):
        """Test if constant timestep produces consistent output"""
        timesteps = torch.tensor([[1], [1]], dtype=torch.float32)
        embeddings = self.model(timesteps)
        self.assertTrue(torch.allclose(embeddings[0], embeddings[1]), "Embeddings for the same timesteps should be "
                                                                      "identical")

    def test_non_zero_gradients(self):
        """Test if the gradients are non-zero"""
        timesteps = torch.tensor([[1], [2], [3]], dtype=torch.float32, requires_grad=True)
        embeddings = self.model(timesteps)
        loss = embeddings.sum()
        loss.backward()
        self.assertTrue(timesteps.grad is not None, "Gradients should not be None")
        self.assertTrue((timesteps.grad != 0).any(), "Gradients should not be all zero")


if __name__ == "__main__":
    unittest.main()
