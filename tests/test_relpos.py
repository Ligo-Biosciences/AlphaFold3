import unittest
import torch
import numpy as np
from src.models.components.relative_position_encoding import RelativePositionEncoding


class TestRelativePositionEncoding(unittest.TestCase):
    def setUp(self):
        self.c_pair = 32
        self.r_max = 32
        self.s_max = 2
        self.rel_pos_encoding = RelativePositionEncoding(self.c_pair, self.r_max, self.s_max)

    def test_initialization(self):
        self.assertEqual(self.rel_pos_encoding.c_pair, self.c_pair)
        self.assertEqual(self.rel_pos_encoding.r_max, self.r_max)
        self.assertEqual(self.rel_pos_encoding.s_max, self.s_max)

    def test_forward_shape(self):
        batch_size = 2
        n_tokens = 10
        features = {
            "residue_index": torch.arange(n_tokens).repeat(batch_size, 1).float(),
            "token_index": torch.arange(n_tokens).repeat(batch_size, 1).float(),
            "asym_id": torch.zeros(batch_size, n_tokens).long(),
            "entity_id": torch.zeros(batch_size, n_tokens).long(),
            "sym_id": torch.zeros(batch_size, n_tokens).long(),
        }
        
        output = self.rel_pos_encoding(features)
        expected_shape = (batch_size, n_tokens, n_tokens, self.c_pair)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_mask(self):
        batch_size = 2
        n_tokens = 10
        features = {
            "residue_index": torch.arange(n_tokens).repeat(batch_size, 1).float(),
            "token_index": torch.arange(n_tokens).repeat(batch_size, 1).float(),
            "asym_id": torch.zeros(batch_size, n_tokens).long(),
            "entity_id": torch.zeros(batch_size, n_tokens).long(),
            "sym_id": torch.zeros(batch_size, n_tokens).long(),
        }
        mask = torch.ones(batch_size, n_tokens)
        mask[:, 5:] = 0  # Mask out the last 5 tokens

        output = self.rel_pos_encoding(features, mask)
        self.assertTrue(torch.all(output[:, 5:, :, :] == 0))
        self.assertTrue(torch.all(output[:, :, 5:, :] == 0))

    def test_encode_method(self):
        feature_tensor = torch.tensor([1, 2, 3, 4])
        condition_tensor = torch.tensor([[True, True, False, False],
                                         [True, True, False, False],
                                         [False, False, True, True],
                                         [False, False, True, True]])
        clamp_max = 2
        device = torch.device("cpu")
        dtype = torch.float32

        result = RelativePositionEncoding._encode(feature_tensor, condition_tensor, clamp_max, device, dtype)
        expected_shape = (4, 4, 2 * clamp_max + 2)
        self.assertEqual(result.shape, expected_shape)

    def test_different_chain_encoding(self):
        batch_size = 1
        n_tokens = 6
        features = {
            "residue_index": torch.arange(n_tokens).unsqueeze(0).float(),
            "token_index": torch.arange(n_tokens).unsqueeze(0).float(),
            "asym_id": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "entity_id": torch.zeros(batch_size, n_tokens).long(),
            "sym_id": torch.zeros(batch_size, n_tokens).long(),
        }

        output = self.rel_pos_encoding(features)
        # Check that tokens in different chains have different encodings
        self.assertFalse(torch.allclose(output[0, 0, 0], output[0, 0, 3]))

if __name__ == '__main__':
    unittest.main()
