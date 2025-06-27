"""TrainArgs: Used for training setup."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type

import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.strategies import Strategy
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection


@dataclass
class TrainArgs:
    """Used for training setup."""

    config_path: str
    loss: nn.Module
    lr: float
    max_epochs: int
    project: str
    run: str
    checkpoints_location: str
    optimizer_type: Optional[Type[optim.Optimizer]] = None
    weight_decay: float = 1e-2
    optimizer_kwargs: Optional[dict[str, Any]] = None
    scheduler_type: Optional[Type[lr_scheduler.LRScheduler]] = None
    scheduler_kwargs: Optional[dict[str, Any]] = None
    lr_scheduler_config: Optional[dict[str, Any]] = None
    train_metrics: Optional[MetricCollection] = None
    val_metrics: Optional[MetricCollection] = None
    test_metrics: Optional[MetricCollection] = None
    checkpoint_path: Optional[str] = None
    load_fn: Optional[Callable[[str], OrderedDict]] = None
    freeze_pretrained: bool = False
    devices: list[int] | str | int = "auto"
    strategy: str | Strategy = "auto"
    callbacks: Optional[list[Callback]] = None
    fast_dev_run: int | bool = False
    trainer_kwargs: Optional[dict[str, Any]] = None
