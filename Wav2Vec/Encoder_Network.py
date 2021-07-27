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

class Encoder(nn.Module):
    def __init__(self, conv_layers, dropout_rate, skip_connections, residual_scale, non_affine_group_norm):
        super(Encoder, self).__init__()

        def block(channel_in, channel_out, k_size, stride):
            return nn.Sequential([nn.Conv1d(channel_in, channel_out, k_size, stride),
                                  nn.Dropout(p=dropout_rate),
                                  Fp32GroupNorm(num_groups=1, dim=channel_out, affine=not non_affine_group_norm),
                                  nn.ReLU(inplace=True)])

        self.conv_layers = nn.ModuleList()
        self.skip_connections = skip_connections
        self.residual_scale = np.sqrt(residual_scale)


        in_d = 1
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.conv_layers = nn.Sequential(*self.conv_layers)

    def forward(self, input):
        # B x T ----> B x C x T
        input = input.unsqueeze(1)

        for conv in self.conv_layers:
            residual = input

            input = conv(input)

            if self.skip_connections and input.shape[1] == residual.shape[1]:
                tsz = input.shape[2]
                r_tsz = residual.shape[2]

                residual = residual[..., :: r_tsz // tsz][..., tsz]
                x = (x + residual) * self.residual_scale

        return x