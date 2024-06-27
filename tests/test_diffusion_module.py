""""""
import torch
import unittest
from src.models.diffusion_module import DiffusionModule
from src.utils.geometry.vector import Vec3Array


class TestDiffusionModule(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.n_atoms = 384 * 4

        self.c_atom = 128
        self.c_atompair = 16
        self.c_token = 768
        self.c_tokenpair = 128
        self.n_tokens = 384
        self.atom_encoder_blocks = 3
        self.atom_encoder_heads = 16
        self.dropout = 0.0
        self.atom_attention_n_queries = 32
        self.atom_attention_n_keys = 128
        self.atom_decoder_blocks = 3
        self.atom_decoder_heads = 16
        self.token_transformer_blocks = 24
        self.token_transformer_heads = 16

        # self.r_max = 32
        # self.s_max = 2
        self.module = DiffusionModule()  # values above are default values
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
        residue_index = torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens))
        # Input features to the model
        self.features = {
            'ref_pos': torch.rand(self.batch_size, self.n_atoms, 3),
            'ref_charge': torch.rand(self.batch_size, self.n_atoms),
            'ref_mask': torch.ones(self.batch_size, self.n_atoms),
            'ref_element': torch.rand(self.batch_size, self.n_atoms, 128),
            'ref_atom_name_chars': torch.randint(0, 2, (self.batch_size, self.n_atoms, 4, 64)),
            'ref_space_uid': residue_index.unsqueeze(-1).expand(self.batch_size, self.n_tokens, 4).reshape(self.batch_size, self.n_tokens * 4),
            'atom_to_token': torch.randint(0, self.n_tokens, (self.batch_size, self.n_atoms)),
            "residue_index": residue_index,
            "token_index": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "asym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "entity_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
            "sym_id": torch.randint(0, self.n_tokens, (self.batch_size, self.n_tokens)),
        }
        self.noisy_atoms = torch.randn(self.batch_size, self.n_atoms, 3)
        self.t = torch.randn(self.batch_size, 1)
        self.s_inputs = torch.randn(self.batch_size, self.n_tokens, self.c_token)
        self.s_trunk = torch.randn(self.batch_size, self.n_tokens, self.c_token)
        self.z_trunk = torch.randn(self.batch_size, self.n_tokens, self.n_tokens, self.c_tokenpair)
        self.sd_data = 16.0  # torch.randn(self.batch_size, 1)  # standard dev of data (bs, 1)

    def test_forward(self):
        output = self.module(noisy_atoms=self.noisy_atoms,  # (bs, n_atoms)
                             timesteps=self.t,
                             features=self.features,  # input feature dict
                             s_inputs=self.s_inputs,  # (bs, n_tokens, c_token)
                             s_trunk=self.s_trunk,  # (bs, n_tokens, c_token)
                             z_trunk=self.z_trunk,  # (bs, n_tokens, n_tokens, c_pair)
                             sd_data=self.sd_data)
        self.assertEqual(output.shape, (self.batch_size, self.n_atoms, 3))

    def test_gradients_not_none(self):
        self.optimizer.zero_grad()
        output = self.module(noisy_atoms=self.noisy_atoms,  # (bs, n_atoms)
                             timesteps=self.t,
                             features=self.features,  # input feature dict
                             s_inputs=self.s_inputs,  # (bs, n_tokens, c_token)
                             s_trunk=self.s_trunk,  # (bs, n_tokens, c_token)
                             z_trunk=self.z_trunk,  # (bs, n_tokens, n_tokens, c_pair)
                             sd_data=self.sd_data)
        output = output.to_tensor()
        loss = torch.mean((output - torch.ones_like(output)) ** 2)
        loss.backward()

        for name, param in self.module.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
