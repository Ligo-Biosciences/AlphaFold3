# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy

import torch
import numpy as np
import unittest
from src.models.components.triangular_attention import TriangleAttention

from tests.config import consts


class TestTriangularAttention(unittest.TestCase):
    def test_shape(self):
        c_z = consts.c_z
        c = 12
        no_heads = 4
        starting = True

        tan = TriangleAttention(c_z, c, no_heads, starting)

        batch_size = consts.batch_size
        n_res = consts.n_res

        x = torch.rand((batch_size, n_res, n_res, c_z))
        shape_before = x.shape
        x = tan(x, chunk_size=None)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
