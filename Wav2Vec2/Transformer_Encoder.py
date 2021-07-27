from typing import List, Tuple
import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from backend import Fp32LayerNorm, SamePad
from Transformer_Sentence_Encoder import TransformerSentenceEncoder


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.embedding_dim

        self.pos_conv = nn.Conv1d(self.embedding_dim,
                                  self.embedding_dim,
                                  kernel_size=args.conv_pos,
                                  padding=args.conv_pos // 2,
                                  groups=args.conv_pos_groups)

        dropout = 0
        std = np.sqrt((4 * (1. - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)

        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.layer = nn.ModuleList(
        [
            TransformerSentenceEncoder(
                embedding_dim = args.embedding_dim,
                ffn_embedding_dim = args.encoder_ffn_embed_dim,
                num_attention_heads = args.encoder_attention_heads,
                dropout = args.dropout,
                attention_dropout = args.attention_dropout,
                activation_dropout = args.activation_dropout,
                activation_fn = args.activation_fn,
                layer_norm_first = args.layer_norm_first
            )
            for _ in range(args.encoder_layers)
        ]
        )

        self.layer_norm_first = args.layer_norm_first

        self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, input, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(input, padding_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, input, padding_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x = index_put(input, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C --> T x B x C

        x = x.transpose(0, 1)

        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weight=False)
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        x = x.transpose(0, 1)

        return x, layer_results

    def max_positions(self):
        return self.args.max_positions

    def update_state_dict_named(self, state_dict, name):
        return state_dict
