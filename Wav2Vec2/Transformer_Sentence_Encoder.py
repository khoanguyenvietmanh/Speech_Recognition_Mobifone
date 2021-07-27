from typing import List, Tuple
import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from backend import Fp32LayerNorm


class TransformerSentenceEncoder(nn.Module):
    def __ini__(self, embedding_dim: float = 768,
                ffn_embedding_dim: float = 3072,
                num_attention_heads: float = 8,
                dropout: float = 0.1,
                attention_dropout: float = 0.1,
                activation_dropout: float = 0.1,
                activation_fn: str = "relu",
                layer_norm_first: bool = False):

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = dropout

        # Initialize self-attetion blocks
        self.self_attn = MultiHeadAttention(
            self.embedding_size,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True
        )

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=self.activation_dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.layer_norm_first = layer_norm_first

        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_size)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

        def forward(
                self,
                x: torch.Tensor,
                self_attn_mask: torch.Tensor = None,
                self_attn_padding_mask: torch.Tensor = None,
                need_weights: bool = False,
                att_args=None,
        ):

            residual = x

            if self.layer_norm_first:
                x = self.self_attn_layer_norm(x)
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                    attn_mask=self_attn_mask,
                )
                x = self.dropout1(x)
                x = residual + x

                residual = x
                x = self.final_layer_norm(x)
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                x = self.dropout3(x)
                x = residual + x
            else:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                )

                x = self.dropout1(x)
                x = residual + x

                x = self.self_attn_layer_norm(x)

                residual = x
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)
                x = self.dropout3(x)
                x = residual + x
                x = self.final_layer_norm(x)

            return x, attn
