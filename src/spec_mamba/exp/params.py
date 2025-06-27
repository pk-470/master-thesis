"""
Params: Dataclass containing all experiment parameters:
TrainerType, ModelType, ModelArgs, DataArgs, TrainArgs.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

from spec_mamba.models import ModelType
from spec_mamba.training import DataArgs, ModelArgs, TrainArgs, TrainModule


@dataclass
class Params:
    """Dataclass containing all experiment parameters."""

    train_module_type: type[TrainModule]
    model_type: type[ModelType]
    model_args: ModelArgs
    data_args: DataArgs
    train_args: TrainArgs

    @staticmethod
    def from_cfg(cfg_filename: str) -> Params:
        cfg_module = f"spec_mamba.exp.cfg.{cfg_filename}"

        if importlib.util.find_spec(cfg_module) is None:
            raise FileNotFoundError(
                f"Configuration file '{cfg_filename}.py' not found in 'spec_mamba/exp/cfg/'."
            )

        return importlib.import_module(cfg_module).PARAMS
