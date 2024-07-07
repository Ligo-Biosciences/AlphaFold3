import unittest
import torch
from src.models.embedders import TemplateEmbedder


class TestTemplateEmbedder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_templates = 2
        self.n_tokens = 64
        self.c_template = 32
        self.c_z = 128

        self.module = TemplateEmbedder(c_template=self.c_template, c_z=self.c_z)

    def test_forward(self):
        features = {
            "template_restype": torch.randn((self.batch_size, self.n_templates, self.n_tokens, 32)),
            "template_pseudo_beta_mask": torch.randn((self.batch_size, self.n_templates, self.n_tokens)),
            "template_backbone_frame_mask": torch.randn((self.batch_size, self.n_templates, self.n_tokens)),
            "template_distogram": torch.randn((self.batch_size, self.n_templates, self.n_tokens, self.n_tokens, 39)),
            "template_unit_vector": torch.randn((self.batch_size, self.n_templates, self.n_tokens, self.n_tokens, 3)),
            "asym_id": torch.ones((self.batch_size, self.n_tokens))
        }
        z = torch.randn((self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
        pair_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens))
        embeddings = self.module(features, z, pair_mask)
        self.assertEqual(embeddings.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_template))
