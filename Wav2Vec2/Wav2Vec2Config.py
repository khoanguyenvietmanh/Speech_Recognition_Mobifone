from dataclasses import dataclass, field
from Base_Dataclass import BaseDataclass
from backend import ChoiceEnum
from typing import List, Tuple, Optional

# mode for extract feature encoder
EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])

# mode for masking
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


ACTIVATION_FUNCTION_CHOICES = ChoiceEnum(["relu", "gelu", "gelu_accurate", "tanh", "linear",])

@dataclass
class Wav2Vec2Config(BaseDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default = "default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
                    "groups in the first conv block, whereas layer_norm has layer norms in "
                    "every block (meant to use with normalize=True)"
        }
    )

    encoder_layers: int = field(
        default = 12, metadata={"help": "number of encoder layers in transformer"}
    )

    encoder_embed_dim: int = field(
        default = 768, metadata={"help": "encoder embedding dimension"}
    )

    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}

    )

    encoder_attention_heads: int = field(
        default=12, metadata={"help": "number of encoder attention heads"}
    )

    ativation_fn: ACTIVATION_FUNCTION_CHOICES = field(
        default="gelu", metadat={"help": "activation function"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for transformer"}
    )

    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )

    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )

    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a transformer layer"}
    )

    dropout_input: float = field(
        default=0.0, metadata={"help": "dropout to apply to the input (after feature extractor)"}
    )

    dropout_features: float = field(
        default=0.0, metadata={"help": "dropout to apply the features (after feature extractor"}
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        }
    )

    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )

    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        }
    )

    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )

    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )

    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )

    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )

    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )

    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_before: bool = False
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )



