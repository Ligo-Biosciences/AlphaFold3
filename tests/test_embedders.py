import unittest
import torch
from src.models.embedders import TemplateEmbedder, InputEmbedder


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


class TestInputEmbedder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.n_atoms = 64
        self.n_tokens = 64
        self.c_token = 384
        self.c_atom = 128
        self.c_atompair = 16
        self.c_z = 128

        self.module = InputEmbedder(
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_trunk_pair=self.c_z
        )

    def test_forward(self):
        features = {
            'ref_pos': torch.randn(self.batch_size, self.n_atoms, 3),
            'ref_charge': torch.randn(self.batch_size, self.n_atoms),
            'ref_mask': torch.ones(self.batch_size, self.n_atoms),
            'ref_element': torch.randn(self.batch_size, self.n_atoms, 128),
            'ref_atom_name_chars': torch.randint(0, 2, (self.batch_size, self.n_atoms, 4, 64)),
            'ref_space_uid': torch.ones((self.batch_size, self.n_atoms)).float(),
            "residue_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "token_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "asym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "entity_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "sym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            'atom_to_token': torch.randint(0, self.n_tokens, (self.batch_size, self.n_atoms)),
            'token_bonds': torch.randint(0, 2, (self.batch_size, self.n_tokens, self.n_tokens)).float()
        }
        token_mask = torch.randint(0, 2, (self.batch_size, self.n_tokens))
        atom_mask = torch.randint(0, 2, (self.batch_size, self.n_atoms))
        s_inputs, s_init, z_init = self.module(features, atom_mask, token_mask)
        self.assertEqual(s_inputs.shape, (self.batch_size, self.n_tokens, self.c_token))
        self.assertEqual(s_init.shape, (self.batch_size, self.n_tokens, self.c_token))
        self.assertEqual(z_init.shape, (self.batch_size, self.n_tokens, self.n_tokens, self.c_z))
