import unittest
import torch
import numpy as np
from prettytable import PrettyTable

from tests.config import consts
from src.models.components.evoformer_pair_stack import (
    EvoformerPairStackBlock,
    EvoformerPairStack
)


class TestEvoformerPairStackBlock(unittest.TestCase):
    def test_shape(self):
        c_s = consts.c_z
        n_heads = 4
        c_hidden = 128
        dropout_rate = 0.25
        block = EvoformerPairStackBlock(c_s, n_heads, c_hidden, dropout_rate)
        batch_size = consts.batch_size
        n_res = consts.n_res

        z = torch.rand((batch_size, n_res, n_res, c_s))
        shape_before = z.shape
        output = block(z)
        shape_after = output.shape

        self.assertTrue(shape_before == shape_after)


class TestEvoformerPairStack(unittest.TestCase):
    def test_shape(self):
        c_s = consts.c_z
        n_heads = 4
        c_hidden = 128
        dropout_rate = 0.25
        n_blocks = 4
        stack = EvoformerPairStack(n_blocks, c_s, n_heads, c_hidden, dropout_rate)
        batch_size = consts.batch_size
        n_res = consts.n_res

        z = torch.rand((batch_size, n_res, n_res, c_s))
        shape_before = z.shape
        output = stack(z)
        shape_after = output.shape

        self.assertTrue(shape_before == shape_after)

    def test_params(self):
        """A method to check the number of parameters in the stack."""
        def count_parameters(model):
            table = PrettyTable(["Modules", "Parameters"])
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
            print(table)
            print(f"Total Trainable Params: {total_params}")
            return total_params

        c_s = consts.c_z
        n_heads = 4
        c_hidden = 128
        dropout_rate = 0.25
        n_blocks = 1
        stack = EvoformerPairStack(n_blocks, c_s, n_heads, c_hidden, dropout_rate)
        count_parameters(stack)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
