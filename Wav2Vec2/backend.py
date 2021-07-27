from typing import List, Tuple
import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from enum import Enum, EnumMeta


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        output = F.group_norm(
            input = input.float(),
            num_groups = self.num_groups,
            weight= self.weight.float() if self.weight is not None else None,
            bias = self.bias.float() if self.bias is not None else None,
            eps = self.eps
        )

        return output.type_as(input)

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_index=None):
        self.deconstruct_index = deconstruct_index

    def foward(self, input):
        if self.deconstruct_index is not None:
            input = input[self.deconstruct_index]

        return input.transpose(-2, -1)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        output = F.layer_norm(
            input = input.float(),
            normalized_shape = self.normalized_shape,
            weight= self.weight.float() if self.weight is not None else None,
            bias = self.bias.float() if self.bias is not None else None,
            eps = self.eps
        )

        return output.type_as(input)


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x



class StrEnumMeta(EnumMeta):
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})