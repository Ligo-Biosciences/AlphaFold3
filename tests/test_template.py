import unittest
import torch
from src.models.template import TemplatePairStack


class TestTemplatePairStack(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_tokens = 64
        self.c_z = 32
        self.no_blocks = 1
        self.c_hidden_mul = 64
        self.c_hidden_pair_att = 16
        self.no_heads_pair = 4
        self.no_heads_single_att = 2
        self.transition_n = 1
        self.pair_dropout = 0.25
        self.module = TemplatePairStack(c_template=self.c_z)

    def test_forward(self):
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        z_out = self.module(z, pair_mask)
        self.assertEqual(z_out.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
