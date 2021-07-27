import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super(Fp32GroupNorm, self).__init__()

    def forward(self, input):
        output = F.group_norm(
            input = input.float(),
            weight = self.weight.float() if self.weight is not None else None,
            bias = self.bias.float() if self.bias is not None else None,
            eps= self.eps
        )

        return output.type_as(input)


class ZeroPad1D(nn.Module):
    def __init__(self, padding_left, padding_right):
        super(ZeroPad1D, self).__init__()
        self.pad_left = padding_left
        self.pad_right = padding_right

    def forward(self, input):
        return F.pad(input, (self.pad_left, self.pad_right))


class Context(nn.Module):
    def __init__(self, embedding_size, conv_layers, dropout_rate, skip_connections, residual_scale, non_affine_group_norm):
        super(Context, self).__init__()

        def block(channel_in, channel_out, k_size, stride):
            ka = k_size // 2

            kb = ka - 1 if k_size % 2 == 0 else ka

            pad = ZeroPad1D(ka + kb, 0)

            return nn.Sequential([pad,
                                  nn.Conv1d(channel_in, channel_out, k_size, stride),
                                  nn.Dropout(p=dropout_rate),
                                  Fp32GroupNorm(num_groups=1, dim=channel_out, affine=not non_affine_group_norm),
                                  nn.ReLU(inplace=True)])


        self.conv_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        self.skip_connections = skip_connections
        self.residual_scale = np.sqrt(residual_scale)

        in_d = embedding_size

        for dim, k, stride in conv_layers:
            if in_d != dim and self.skip_connections:
                self.residual_proj.append(nn.Conv1d(in_d, dim, k, stride))
            else:
                self.residual_proj.append(None)

            self.conv_layers.append(block(in_d, dim, k, stride))

            in_d = dim

        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.residual_proj = nn.Sequential(*self.residual_proj)

    def forward(self, input):

        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = input
            input = conv(input)

            if self.skip_connections:
                if rproj is not None:
                    residual = rproj(residual)
                input = (input + residual) * self.residual_scale

        return input

