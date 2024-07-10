import unittest
import torch
from src.models.heads import ConfidenceHead


class TestConfidenceHead(unittest.TestCase):
    def setUp(self):
        self.c_s = 128
        self.c_z = 256
        self.n_tokens = 512
        self.batch_size = 2
        self.no_blocks = 1

        self.module = ConfidenceHead(self.c_s, self.c_z, no_blocks=self.no_blocks)

    def test_forward(self):
        s_inputs = torch.randn((self.batch_size, self.n_tokens, self.c_s))
        s = torch.randn((self.batch_size, self.n_tokens, self.c_s))
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        x_repr = torch.randn((self.batch_size, self.n_tokens, 3))  # (bs, n_tokens, 3)
        single_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        output = self.module(s_inputs, s, z, x_repr, single_mask=single_mask, pair_mask=pair_mask)
        self.assertEqual(
            output["logits_plddt"].shape,
            (self.batch_size, self.n_tokens, self.module.no_bins_plddt)
        )
        self.assertEqual(
            output["logits_pae"].shape,
            (self.batch_size, self.n_tokens, self.n_tokens, self.module.no_bins_pae)
        )
        self.assertEqual(
            output["logits_pde"].shape,
            (self.batch_size, self.n_tokens, self.n_tokens, self.module.no_bins_pde)
        )
        self.assertEqual(
            output["logits_p_resolved"].shape,
            (self.batch_size, self.n_tokens, 2)
        )

