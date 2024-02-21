# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.models.components.primitives import Linear


class StructureTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init="relu")
        self.linear_2 = Linear(self.c, self.c, init="relu")
        self.linear_3 = Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial
        return s


class StructureTransition(nn.Module):
    def __init__(self,
                 c: int,
                 num_layers: int = 1,
                 dropout_rate: float = 0.1):
        """
        Args:
            c: the number of channels in the input and output tensors.
            num_layers: int, the number of structure transition layers.
            dropout_rate: float, the dropout rate
        """
        super(StructureTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = StructureTransitionLayer(self.c)
            self.layers.append(layer)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.c)

    def forward(self, s):
        for layer in self.layers:
            s = layer(s)
        s = self.dropout(s)
        s = self.layer_norm(s)
        return s
