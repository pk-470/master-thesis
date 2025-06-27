"""Classifier modules."""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from spec_mamba.models.common.base_model import BaseModel


class Classifier(nn.Sequential):
    """Classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_features: Optional[int] = None,
        norm_layer: Optional[type[nn.Module]] = None,
        activation_layer: Optional[nn.Module] = nn.ReLU(),
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        features = hidden_features if hidden_features is not None else in_features
        normalization_layer = (
            norm_layer(features) if norm_layer is not None else nn.Identity()
        )
        activation_layer = (
            activation_layer if activation_layer is not None else nn.Identity()
        )
        dropout_layer = nn.Dropout(dropout)
        layers = (
            [nn.Linear(in_features, hidden_features, bias=bias)]
            if hidden_features is not None
            else []
        )
        layers += [
            normalization_layer,
            activation_layer,
            dropout_layer,
            nn.Linear(features, num_classes, bias=bias),
        ]
        super().__init__(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseCLF(nn.Module):
    """Trainable classifier base."""

    backbone_type: type[BaseModel]
    backbone: BaseModel

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._process_args(**kwargs)
        self.backbone = self.backbone_type(**self.backbone_args)
        self.classifier = Classifier(**self.clf_args)

        if self.backbone.use_pred_head:
            raise ValueError("Set use_pred_head=False when initializing classifier.")
        if self.backbone.output_type not in ("cls", "last", "mean", "max"):
            raise ValueError(
                "For classification choose one of the reduced output types: 'cls', 'last', 'mean', 'max'."
            )

    @property
    def no_weight_decay(self) -> tuple[str, ...]:
        return ("cls_token", "mask_token", "pos_embed")

    def _process_args(self, **kwargs) -> None:
        clf_args = {}
        clf_args["in_features"] = kwargs["embed_dim"]
        clf_args["num_classes"] = kwargs.pop("num_classes")
        for key in (
            "hidden_features",
            "norm_layer",
            "activation_layer",
            "bias",
            "dropout",
        ):
            if (clf_key := f"clf_{key}") in kwargs:
                clf_args[key] = kwargs.pop(clf_key)

        self.backbone_args = kwargs
        self.clf_args = clf_args

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.backbone.forward_features(x, mask_ratio=0.0)
        logits = self.classifier(x)

        return logits
