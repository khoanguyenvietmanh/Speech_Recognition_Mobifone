from dataclasses import dataclass, field
import logging
import math
from typing import Optional, Tuple
from omegaconf import II
import sys
import logging
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import logging
from configs import FairseqDataclass
from constants import ChoiceEnum
from Base_Model import BaseModel
from Context_Network import Context
from Encoder_Network import Encoder
from Wav2Vec_Prediction_Model import Wav2VecPredictionsModel

logger = logging.getLogger(__name__)

AGGREGATOR_CHOICES = ChoiceEnum(["cnn", "gru"])
PROJECT_FEATURES_CHOICES = ChoiceEnum(["none", "same", "new"])
ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu"])
VQ_TYPE_CHOICES = ChoiceEnum(["none", "gumbel", "kmeans"])


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


@dataclass
class Wav2VecConfig(FairseqDataclass):
    prediction_steps: int = field(
        default=12, metadata={"help": "number of steps ahead to predict"}
    )
    sample_distance: Optional[int] = field(
        default=None,
        metadata={
            "help": "sample distance from target. does not work properly with cross-sampling"
        },
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "num of cross sampled negatives"}
    )
    num_negatives: int = field(
        default=10, metadata={"help": "num of sampled negatives"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]",
        metadata={
            "help": "convolutional feature extraction layers [(dim, kernel_size, stride), ...]"
        },
    )
    conv_aggregator_layers: str = field(
        default="[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]",
        metadata={
            "help": "convolutional aggregator layers [(dim, kernel_size, stride), ...]"
        },
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout to apply within the model"}
    )
    dropout_features: float = field(
        default=0.0, metadata={"help": "dropout to apply to the features"}
    )
    dropout_agg: float = field(
        default=0.0, metadata={"help": "dropout to apply after aggregation step"}
    )
    aggregator: AGGREGATOR_CHOICES = field(
        default="cnn", metadata={"help": "type of aggregator to use"}
    )
    gru_dim: int = field(default=512, metadata={"help": "GRU dimensionality"})
    no_conv_bias: bool = field(
        default=False, metadata={"help": "if set, does not learn bias for conv layers"}
    )
    agg_zero_pad: bool = field(
        default=False,
        metadata={"help": "if set, zero pads in aggregator instead of repl pad"},
    )
    skip_connections_feat: bool = field(
        default=False,
        metadata={"help": "if set, adds skip connections to the feature extractor"},
    )
    skip_connections_agg: bool = field(
        default=True,
        metadata={"help": "if set, adds skip connections to the aggregator"},
    )
    residual_scale: float = field(
        default=0.5, metadata={"help": "scales residual by sqrt(value)"}
    )
    log_compression: bool = field(
        default=True,
        metadata={"help": "if set, adds a log compression to feature extractor"},
    )
    balanced_classes: bool = field(
        default=False,
        metadata={"help": "if set, loss is scaled to balance for number of negatives"},
    )
    project_features: PROJECT_FEATURES_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, features are projected using the (same or new) aggregator"
        },
    )
    non_affine_group_norm: bool = field(
        default=False, metadata={"help": "if set, group norm is not affine"}
    )
    offset: str = field(
        default="auto",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )
    activation: ACTIVATION_CHOICES = field(
        default="relu",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )
    vq_type: VQ_TYPE_CHOICES = field(
        default="none", metadata={"help": "which type of quantizer to use"}
    )
    vq_vars: int = field(
        default=320,
        metadata={"help": "project to this many vector quantized variables per group"},
    )
    vq_groups: int = field(
        default=2, metadata={"help": "number of groups of latent variables"}
    )
    vq_dim: int = field(
        default=0,
        metadata={
            "help": "uses this dimensionality for quantized vectors. 0 to use model dim // groups"
        },
    )
    vq_depth: int = field(
        default=1, metadata={"help": "number of layers for vq weight projection"}
    )
    combine_groups: bool = field(
        default=False, metadata={"help": "if set, variables are shared among groups"}
    )
    vq_temp: Tuple[float, float, float] = field(
        default=(2.0, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling with gumbel softmax. should be a tuple of 3 values (start, end, decay)"
        },
    )
    vq_gamma: float = field(
        default=0.25,
        metadata={"help": "gamma parameter for kmeans style vector quantization"},
    )
    infonce: bool = II("criterion.infonce")


class Wav2VecModel(BaseModel):
    @classmethod
    def build_model(cls, cfg: Wav2VecConfig):
        """Build a new model instance."""
        model = Wav2VecModel(cfg)
        logger.info(model)
        return model

    def __init__(self, cfg: Wav2VecConfig):
        super().__init__()

        self.prediction_steps = cfg.prediction_steps
        offset = cfg.offset

        if cfg.activation == "relu":
            activation = nn.ReLU()
        elif cfg.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + cfg.activation)

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.feature_extractor = Encoder(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            log_compression=cfg.log_compression,
            skip_connections=cfg.skip_connections_feat,
            residual_scale=cfg.residual_scale,
            non_affine_group_norm=cfg.non_affine_group_norm,
            activation=activation,
        )
        embed = feature_enc_layers[-1][0]

        # self.vector_quantizer = None
        # if cfg.vq_type == "gumbel":
        #     self.vector_quantizer = GumbelVectorQuantizer(
        #         dim=embed,
        #         num_vars=cfg.vq_vars,
        #         temp=cfg.vq_temp,
        #         groups=cfg.vq_groups,
        #         combine_groups=cfg.combine_groups,
        #         vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
        #         time_first=False,
        #         activation=activation,
        #         weight_proj_depth=cfg.vq_depth,
        #         weight_proj_factor=2,
        #     )
        # elif cfg.vq_type == "kmeans":
        #     self.vector_quantizer = KmeansVectorQuantizer(
        #         dim=embed,
        #         num_vars=cfg.vq_vars,
        #         groups=cfg.vq_groups,
        #         combine_groups=cfg.combine_groups,
        #         vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,
        #         time_first=False,
        #         gamma=cfg.vq_gamma,
        #     )
        # else:
        #     assert (
        #         cfg.vq_type == "none" or cfg.vq_type is None
        #     ), "Unknown quantizer type"

        if cfg.offset == "auto":
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)

        offset = int(offset)

        def make_aggregator():
            if cfg.aggregator == "cnn":
                agg_layers = eval(cfg.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                feature_aggregator = Context(
                    conv_layers=agg_layers,
                    embed=embed,
                    dropout=cfg.dropout,
                    skip_connections=cfg.skip_connections_agg,
                    residual_scale=cfg.residual_scale,
                    non_affine_group_norm=cfg.non_affine_group_norm,
                    conv_bias=not cfg.no_conv_bias,
                    zero_pad=cfg.agg_zero_pad,
                    activation=activation,
                )
            elif cfg.aggregator == "gru":
                agg_dim = cfg.gru_dim
                feature_aggregator = nn.Sequential(
                    TransposeLast(),
                    nn.GRU(
                        input_size=embed,
                        hidden_size=agg_dim,
                        num_layers=1,
                        dropout=cfg.dropout,
                    ),
                    TransposeLast(deconstruct_idx=0),
                )
            else:
                raise Exception("unknown aggregator type " + cfg.aggregator)

            return feature_aggregator, agg_dim

        self.feature_aggregator, agg_dim = make_aggregator()

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,
            out_dim=embed,
            prediction_steps=cfg.prediction_steps,
            n_negatives=cfg.num_negatives,
            cross_sample_negatives=cfg.cross_sample_negatives,
            sample_distance=cfg.sample_distance,
            dropout=cfg.dropout,
            offset=offset,
            balanced_classes=cfg.balanced_classes,
            infonce=cfg.infonce,
        )

        self.dropout_feats = nn.Dropout(p=cfg.dropout_features)
        self.dropout_agg = nn.Dropout(p=cfg.dropout_agg)

        if cfg.project_features == "none":
            self.project_features = None
        elif cfg.project_features == "same":
            self.project_features = self.feature_aggregator
        elif cfg.project_features == "new":
            self.project_features, _ = make_aggregator()

    def forward(self, source):
        result = {}

        features = self.feature_extractor(source)
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)
            features = q_res["x"]
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        x = self.dropout_feats(features)
        x = self.feature_aggregator(x)
        x = self.dropout_agg(x)

        if self.project_features is not None:
            features = self.project_features(features)
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """Maximum length supported by the model."""
        return sys.maxsize

    def get_logits(self, net_output):
        logits = net_output["cpc_logits"]
        return logits

    def get_targets(self, sample, net_output):
        t = net_output["cpc_targets"]
        if isinstance(t, tuple):
            t = t[0]
        return t.contiguous()

    def get_target_weights(self, targets, net_output):
        targets = net_output["cpc_targets"]
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]
        return None

    def get_extra_losses(self, net_output):
        loss = None
        if "prob_perplexity" in net_output:
            loss = net_output["num_vars"] - net_output["prob_perplexity"]
        elif "kmeans_loss" in net_output:
            loss = net_output["kmeans_loss"]

        return loss