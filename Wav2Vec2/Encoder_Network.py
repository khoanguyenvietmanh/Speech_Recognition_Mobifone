from typing import List, Tuple
import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from backend import TransposeLast, Fp32LayerNorm, Fp32GroupNorm


class Encoder(nn.Module):
    def __init__(self,
                 conv_layers: List[Tuple[int, int, int]],
                 dropout_rate: float = 0.0,
                 mode: str = "default",
                 conv_bias: bool = False):

        assert mode in {"default", "layer_norm"}

        def block(in_channel, out_channel, k_size, stride, is_layer_norm=False, is_group_norm=False, conv_bia=False):
            if is_layer_norm:
                return nn.Sequential([
                    nn.Conv1d(in_channel, out_channel, k_size, stride),
                    nn.Dropout(p=dropout_rate),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(out_channel, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU()
                ])
            elif is_group_norm:
                return nn.Sequential([
                    nn.Conv1d(in_channel, out_channel, k_size, stride),
                    nn.Dropout(p=dropout_rate),
                    Fp32GroupNorm(out_channel, out_channel, affine=True),
                    nn.GELU()
                ])
            else:
                return nn.Sequential([
                    nn.Conv1d(in_channel, out_channel, k_size, stride),
                    nn.Dropout(p=dropout_rate),
                    nn.GELU()
                ])

        self.conv_layers = nn.ModuleList()
        in_d = 1
        for id, conv in enumerate(conv_layers):
            assert len(conv) == 3, "Invalid conv definition" + str(conv)
            dim, k_size, stride = conv
            self.conv_layers.append(
                block(in_d,
                      dim,
                      k_size,
                      stride,
                      is_layer_norm= mode=="layer_norm",
                      is_group_norm= mode=="default" and id == 0,
                      conv_bias=conv_bias))

            in_d = dim

    def forward(self, input):
        input = input.squeeze(1)

        for conv in self.conv_layers:
            input = conv(input)

        return input
