"""Callback that saves the configuration of the model as an artifact to the WandbLogger."""

import glob
import re
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from spec_mamba.training.checkpoint import save_torch_weights_from_ckpt


def format_artifact_name(run_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", run_name)


class SaveConfigCallback(Callback):
    """Saves the configuration of the model as an artifact to the WandbLogger."""

    def __init__(
        self, config_path: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        self.config_path = config_path
        self.metadata = metadata

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if isinstance(trainer.logger, WandbLogger):
            run: Run = trainer.logger.experiment
            artifact = wandb.Artifact(
                name=format_artifact_name(f"config_{run.name}"),
                type="code",
                description="Training configuration file.",
                metadata=self.metadata,
            )
            artifact.add_file(self.config_path)
            run.log_artifact(artifact)


class SaveTorchWeights(Callback):
    """Save only the weights of the model."""

    def __init__(self, checkpoints_dir: str, log: bool = True, **kwargs: Any) -> None:
        self.checkpoints_dir = checkpoints_dir
        self.log = log
        self.kwargs = kwargs

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for path in glob.glob(f"{self.checkpoints_dir}/*.ckpt"):
            weights_path = save_torch_weights_from_ckpt(
                train_module_type=type(pl_module),
                checkpoint_path=path,
                **self.kwargs,
            )

            if self.log:
                if isinstance(trainer.logger, WandbLogger):
                    run: Run = trainer.logger.experiment
                    artifact = wandb.Artifact(
                        name=format_artifact_name(Path(weights_path).stem),
                        type="weights",
                        description=f"{pl_module.model_name} weights.",
                    )
                    artifact.add_file(weights_path)
                    run.log_artifact(artifact)


class TemperatureScheduler(Callback):
    def __init__(
        self,
        total_steps: int,
        start: float = 1.0,
        end: float = 0.1,
    ):
        super().__init__()
        self.start = start
        self.end = end
        self.total_steps = total_steps

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step < self.total_steps:
            frac = step / self.total_steps
            if not hasattr(pl_module.loss, "temperature"):
                raise ValueError(
                    "The model's loss function must have a 'temperature' attribute."
                )
            pl_module.loss.temperature = self.start + frac * (self.end - self.start)
            pl_module.log(
                "temperature",
                pl_module.loss.temperature,
                on_step=True,
                on_epoch=False,
            )
