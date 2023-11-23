import unittest
import torch
from prettytable import PrettyTable

from tests.config import consts
from src.models.structure_net import StructureNet
from src.utils.rigid_utils import Rigids


class TestStructureNet(unittest.TestCase):
    def test_shape(self):
        c_s = 128
        n_heads = 4
        c_hidden = 128
        dropout_rate = 0.25
        n_blocks = 4
        net = StructureNet(c_s=128, c_z=128)
        batch_size = consts.batch_size
        n_res = consts.n_res
        s = torch.rand((batch_size, n_res, c_s))
        z = torch.rand((batch_size, n_res, n_res, c_s))
        transforms = Rigids.identity((2, 11))

        shape_before = transforms.shape
        mask = torch.ones((batch_size, n_res))
        output = net(s, z, transforms, mask)
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

        net = StructureNet(c_s=128, c_z=128)
        count_parameters(net)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
