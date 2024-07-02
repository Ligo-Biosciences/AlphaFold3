"""Tests for the MSAModule."""
import torch
import unittest
from src.models.msa_module import MSAPairWeightedAveraging, MSAModuleBlock


class TestMSAPairWeightedAveraging(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_tokens = 384
        self.n_seq = 3
        self.c_msa = 64
        self.c_z = 128
        self.c_hidden = 32
        self.no_heads = 8
        self.module = MSAPairWeightedAveraging(self.c_msa, self.c_z, self.c_hidden, self.no_heads)

    def test_forward(self):
        m = torch.randn((self.batch_size, self.n_seq, self.n_tokens, self.c_msa))
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        msa_mask = torch.randint(0, 2, (self.batch_size, self.n_seq, self.n_tokens))
        z_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        output = self.module(m, z, msa_mask, z_mask)

        self.assertEqual(output.shape, (self.batch_size, self.n_seq, self.n_tokens, self.c_msa))


class TestMSAModuleBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_tokens = 384
        self.n_seq = 3
        self.c_msa = 64
        self.c_z = 128
        self.c_hidden = 32
        self.no_heads = 8
        self.module = MSAModuleBlock(c_msa=self.c_msa, c_z=self.c_z, c_hidden=self.c_hidden)

    def test_forward(self):
        m = torch.randn((self.batch_size, self.n_seq, self.n_tokens, self.c_msa))
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        msa_mask = torch.randint(0, 2, (self.batch_size, self.n_seq, self.n_tokens))
        z_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        m_out, z_out = self.module(m, z, msa_mask, z_mask)
        self.assertEqual(m_out.shape, (self.batch_size, self.n_seq, self.n_tokens, self.c_msa))
        self.assertEqual(z_out.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_z))


