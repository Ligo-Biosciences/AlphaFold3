import unittest
import torch
from src.diffusion.conditioning import FourierEmbedding, RelativePositionEncoding, DiffusionConditioning
from src.models.components.primitives import Linear


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


class TestRelativePositionEncoding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_tokens = 384

        self.c_pair = 64
        self.r_max = 32
        self.s_max = 2
        self.module = RelativePositionEncoding(self.c_pair, self.r_max, self.s_max)

    def test_forward(self):
        features = {
            "residue_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "token_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "asym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "entity_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "sym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
        }

        output = self.module(features)
        self.assertEqual(output.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_pair))
        self.assertTrue(torch.is_tensor(output))


class TestDiffusionConditioning(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_tokens = 384

        self.c_token = 128
        self.c_pair = 64
        # self.r_max = 32
        # self.s_max = 2
        self.module = DiffusionConditioning(self.c_token, self.c_pair)

    def test_forward(self):
        features = {
            "residue_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "token_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "asym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "entity_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "sym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
        }
        t = torch.randn(self.batch_size, 1)
        s_inputs = torch.randn(self.batch_size, self.n_tokens, self.c_token)
        s_trunk = torch.randn(self.batch_size, self.n_tokens, self.c_token)
        z_trunk = torch.randn(self.batch_size, self.n_tokens, self.n_tokens, self.c_pair)
        sd_data = torch.randn(self.batch_size, 1)  # standard dev of data (bs, 1)

        output = self.module(t, features, s_inputs, s_trunk, z_trunk, sd_data)
        self.assertEqual(output[0].shape, (self.batch_size, self.n_tokens, self.c_token))
        self.assertEqual(output[1].shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_pair))


if __name__ == "__main__":
    unittest.main()
