"""Dataclasses used to initialize the models."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch.nn as nn

from spec_mamba.training.args.base_args import BaseArgs


@dataclass
class ModelArgs(BaseArgs):
    """Used to initialize the models."""


@dataclass
class BaseModelArgs(ModelArgs):
    """Common model arguments."""

    spec_size: tuple[int, int] = (80, 200)
    patch_size: tuple[int, int] = (16, 4)
    channels: int = 1
    embed_dim: int = 192
    depth: int = 12
    cls_position: Literal["none", "start", "middle", "end", "double"] = "middle"
    output_type: Literal["cls", "last", "mean", "max", "emb", "full"] = "cls"
    masking_mode: Literal["unstructured", "timestep"] = "unstructured"
    mask_token_type: Literal["learned", "noise", "zeros"] = "learned"
    mask_ratio: float = 0.5
    drop_path_rate: float = 0.0
    pos_drop_rate: float = 0.0
    head_drop_rate: float = 0.0
    norm_epsilon: float = 1e-5
    use_rms_norm: bool = False
    use_pred_head: bool = True
    transpose: bool = False
    device: Any = None
    dtype: Any = None


@dataclass
class BaseCLFArgs(ModelArgs):
    """Base dataclass for classifiers."""

    use_pred_head: bool = False
    num_classes: int = 2
    clf_hidden_features: Optional[int] = None
    clf_norm_layer: Optional[type[nn.Module]] = None
    clf_activation_layer: Optional[nn.Module] = nn.ReLU()
    clf_bias: bool = True
    clf_dropout: float = 0.3


@dataclass
class AudioMambaArgs(BaseModelArgs):
    """Used to initialize AudioMamba models."""

    # AudioMamba specific arguments
    bi_mamba_type: Literal["none", "v1", "v2"] = "v1"
    ssm_cfg: Optional[dict[str, Any]] = None
    initializer_cfg: Optional[dict[str, Any]] = None
    fused_add_norm: bool = True
    residual_in_fp32: bool = True
    init_layer_scale: Optional[int] = None


@dataclass
class AudioMambaCLFArgs(BaseCLFArgs, AudioMambaArgs):
    """Used to initialize AudioMambaCLF models."""


@dataclass
class SSASTArgs(BaseModelArgs):
    """Used to initialize SSAST models."""

    # SSAST specific arguments
    num_heads: int = 12
    mlp_ratio: int = 4


@dataclass
class SSASTCLFArgs(BaseCLFArgs, SSASTArgs):
    """Used to initialize SSASTCLF models."""
