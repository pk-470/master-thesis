"""Checkpoint utilities."""

import glob
import os
from collections import OrderedDict
from typing import Any, Callable, Literal, Optional

import lightning.pytorch as pl
import torch

from spec_mamba.models import ModelTypeVar


def get_checkpoint_path(
    checkpoints_location: str,
    model_name: str,
    project: str,
    run: str,
    mode: Literal["best", "last"] = "best",
    weights_only: bool = False,
) -> str:
    """Create the checkpoint path from the model name, project, run, checkpoint mode (best/last)."""
    checkpoint_dir = os.path.join(checkpoints_location, model_name, project, run)
    glob_pattern = f"{mode}*.ckpt" if not weights_only else f"weights_{mode}*.pt"
    ckpts = glob.glob(os.path.join(checkpoint_dir, glob_pattern))
    if not ckpts:
        raise FileNotFoundError(f"No {mode} checkpoint found in {checkpoint_dir}")

    return ckpts[-1]


def save_torch_weights_from_ckpt(
    train_module_type: type[pl.LightningModule], checkpoint_path: str, **kwargs: Any
) -> str:
    """Save only the weights of the model."""
    pl_model = train_module_type.load_from_checkpoint(checkpoint_path, **kwargs)
    weights_path = os.path.join(
        os.path.dirname(checkpoint_path),
        "weights_" + os.path.basename(checkpoint_path).replace(".ckpt", ".pt"),
    )
    torch.save(pl_model.model.state_dict(), weights_path)

    return weights_path


def load_state_dict_from_checkpoint(checkpoint_path: str) -> OrderedDict:
    """Load a state dict from a checkpoint path."""
    if checkpoint_path.endswith(".ckpt"):
        return OrderedDict(
            {
                k.replace("model.", ""): v
                for k, v in torch.load(
                    checkpoint_path, weights_only=False, map_location="cpu"
                )["state_dict"].items()
                if k.startswith("model.")
            }
        )
    if checkpoint_path.endswith(".pt"):
        return torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")


def load_backbone_from_checkpoint(checkpoint_path: str) -> OrderedDict:
    """Load a backbone state dict from a foundation checkpoint path."""
    state_dict = load_state_dict_from_checkpoint(checkpoint_path)

    if checkpoint_path.endswith(".ckpt"):
        rename = lambda k: "backbone." + k.replace("model.", "")
    else:
        rename = lambda k: f"backbone.{k}"

    return OrderedDict({rename(k): v for k, v in state_dict.items()})


def load_and_freeze_state_dict(
    model: ModelTypeVar,
    checkpoint_path: str,
    strict: bool = True,
    freeze_pretrained: bool = False,
    load_fn: Optional[Callable[[str], OrderedDict]] = None,
) -> ModelTypeVar:
    """Load and optionally freeze a pretrained state dict."""
    load_fn = load_fn if load_fn is not None else load_state_dict_from_checkpoint
    pretrained_state_dict = load_fn(checkpoint_path)
    common_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if name in pretrained_state_dict
    )
    if common_params == 0:
        raise RuntimeError(f"No parameters loaded from '{checkpoint_path}'.")
    print(f"Loading {common_params:,} parameters from '{checkpoint_path}'.")

    model.load_state_dict(pretrained_state_dict, strict=strict)

    if freeze_pretrained:
        for name, param in model.named_parameters():
            if name in pretrained_state_dict:
                param.requires_grad = False

    return model
