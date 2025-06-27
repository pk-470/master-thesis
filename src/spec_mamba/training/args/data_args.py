"""Dataclasses used to initialize the datasets and the PyTorch dataloader."""

from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn
from numpy.typing import DTypeLike

from spec_mamba.data.dataset import *
from spec_mamba.training.args.base_args import BaseArgs


@dataclass
class DataArgs(BaseArgs):
    """Base dataclass used to initialize the datasets and the PyTorch dataloader."""

    data_location: str
    splits_dir: str
    processor_type: type[WavToSpecProcessor | SpecProcessor]
    processor_kwargs: Optional[dict[str, Any]] = None
    target_fn: Optional[Callable[[Tensor], Tensor]] = None
    labels: Optional[str | list[str]] = None
    labels_dtype: Optional[DTypeLike] = None
    batch_size: int = 64
    num_workers: int = 16
    train_transform: Optional[nn.Module] = None
    train_batch_transform: Optional[nn.Module] = None
    val_transform: Optional[nn.Module] = None
    val_batch_transform: Optional[nn.Module] = None
    test_transform: Optional[nn.Module] = None
    test_batch_transform: Optional[nn.Module] = None
    dataloader_kwargs: Optional[dict[str, Any]] = None
