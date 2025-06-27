"""TrainModule: PyTorch Lightning wrapper for the training logic."""

import os
import random
from typing import Any, Literal, Optional, Sequence, Type

import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch import Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from spec_mamba.data import GF23Dataset, SpecBatch, get_split_df
from spec_mamba.models import ModelType
from spec_mamba.training.args import DataArgs, ModelArgs, TrainArgs
from spec_mamba.training.callbacks import *
from spec_mamba.training.checkpoint import load_and_freeze_state_dict


class TrainModuleOut(SpecBatch):
    """Output of the forward method of MambaTrainer."""

    logits: Tensor | Sequence[Tensor]


class TrainModule(pl.LightningModule):
    """PyTorch Lightning wrapper for the training logic."""

    def __init__(
        self,
        model_type: Type[ModelType],
        model_args: ModelArgs,
        data_args: DataArgs,
        train_args: Optional[TrainArgs] = None,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_args = model_args
        self.data_args = data_args
        self.train_args = train_args

        self.model = self._init_model()
        self.loss = self._init_loss()
        self.train_metrics = self._init_metrics("train")
        self.val_metrics = self._init_metrics("val")
        self.test_metrics = self._init_metrics("test")

        self.save_hyperparameters(
            ignore=[
                "train_args",
                "model",
                "loss",
                "train_metrics",
                "val_metrics",
                "test_metrics",
            ]
        )

    @property
    def model_name(self) -> str:
        """Class name of model as a string."""
        return type(self.model).__name__

    def _init_model(
        self,
    ) -> ModelType:
        model = self.model_type(**self.model_args)

        if (self.train_args is not None) and (
            self.train_args.checkpoint_path is not None
        ):
            model = load_and_freeze_state_dict(
                model,
                self.train_args.checkpoint_path,
                strict=True,
                freeze_pretrained=self.train_args.freeze_pretrained,
                load_fn=self.train_args.load_fn,
            )

        return model

    def _init_loss(self) -> Optional[nn.Module]:
        if self.train_args is not None:
            return self.train_args.loss
        return None

    def _init_metrics(
        self, mode: Literal["train", "val", "test"]
    ) -> Optional[MetricCollection]:
        if self.train_args is not None:
            metrics: Optional[MetricCollection] = getattr(
                self.train_args, f"{mode}_metrics"
            )
            return metrics.clone(prefix=f"{mode}_") if metrics is not None else None
        return None

    def _get_dataset(self, mode: Literal["train", "val", "test"]) -> GF23Dataset:
        dataframe = get_split_df(self.data_args.splits_dir, mode)
        transform: Optional[nn.Module] = getattr(self.data_args, f"{mode}_transform")
        processor_kwargs = (
            {}
            if self.data_args.processor_kwargs is None
            else self.data_args.processor_kwargs
        )
        processor = self.data_args.processor_type(
            location=self.data_args.data_location,
            dataframe=dataframe,
            spec_transform=transform,
            **processor_kwargs,
        )
        dataset = GF23Dataset(
            processor=processor,
            target_fn=self.data_args.target_fn,
            labels=self.data_args.labels,
            labels_dtype=self.data_args.labels_dtype,
        )

        return dataset

    def _get_dataloader(self, mode: Literal["train", "val", "test"]) -> DataLoader:
        dataset = self._get_dataset(mode)
        dataloader_kwargs = (
            {}
            if self.data_args.dataloader_kwargs is None
            else self.data_args.dataloader_kwargs
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.data_args.batch_size,
            shuffle=(mode == "train"),
            num_workers=self.data_args.num_workers,
            # persistent_workers=True,
            prefetch_factor=4,
            # pin_memory=True,
            **dataloader_kwargs,
        )

        return dataloader

    def _get_parameter_groups(self) -> list[dict[str, Any]]:
        if self.train_args is None:
            raise ValueError("No TrainArgs provided")

        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if (
                any(nd in name for nd in self.model.no_weight_decay)
                and param.requires_grad
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        parameter_groups = [
            {"params": decay_params, "weight_decay": self.train_args.weight_decay}
        ]

        if no_decay_params:
            parameter_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        return parameter_groups

    def _apply_batch_transform(
        self, mode: Literal["train", "val", "test"], batch: SpecBatch
    ) -> SpecBatch:
        batch_transform: Optional[nn.Module] = getattr(
            self.data_args, f"{mode}_batch_transform"
        )
        if batch_transform is not None:
            batch = batch_transform(batch)

        return batch

    def _forward(
        self, mode: Literal["train", "val", "test"], batch: SpecBatch
    ) -> dict[str, Tensor]:
        if ("target" not in batch) or (batch["target"] is None):
            raise ValueError("No target provided.")
        assert self.loss is not None

        x = batch["spectrogram"]
        target = batch["target"]

        logits = self.model(x)
        loss = self.loss(logits, target)

        metrics: Optional[MetricCollection] = getattr(self, f"{mode}_metrics")
        if metrics is not None:
            metrics.update(logits, target)

        return {"loss": loss}

    def _log_losses(
        self,
        mode: Literal["train", "val", "test"],
        losses: dict[str, Tensor],
        batch_size: int,
    ) -> None:
        if self.train_args is None:
            raise ValueError("No TrainArgs provided")

        for loss_type, loss in losses.items():
            self.log(
                f"{mode}_{loss_type}",
                loss,
                on_epoch=True,
                batch_size=batch_size,
                prog_bar=(loss_type == "loss"),
                on_step=(mode == "train"),
                sync_dist=(self.trainer.world_size > 1),
            )

    def _log_metrics(self, mode: Literal["train", "val", "test"]) -> None:
        if self.train_args is None:
            raise ValueError("No TrainArgs provided")

        metrics: Optional[MetricCollection] = getattr(self, f"{mode}_metrics")
        if metrics is not None:
            computed = metrics.compute()
            unpacked_metrics = {}
            for metric_name, metric_value in computed.items():
                assert isinstance(metric_value, Tensor)
                if metric_value.numel() > 1:
                    for idx, val in enumerate(metric_value.view(-1)):
                        unpacked_metrics[f"{metric_name}_{idx}"] = val.item()
                else:
                    unpacked_metrics[metric_name] = metric_value.item()

            self.log_dict(
                unpacked_metrics,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                sync_dist=(self.trainer.world_size > 1),
            )
            metrics.reset()

    def _step(self, mode: Literal["train", "val", "test"], batch: SpecBatch) -> Tensor:
        if self.train_args is None:
            raise ValueError("No TrainArgs provided")

        batch = self._apply_batch_transform(mode, batch)
        losses = self._forward(mode, batch)

        assert "loss" in losses
        self._log_losses(mode, losses, batch_size=batch["spectrogram"].shape[0])

        return losses["loss"]

    def forward(self, batch: SpecBatch) -> TrainModuleOut:
        x = batch["spectrogram"]
        logits = self.model(x)

        return TrainModuleOut(logits=logits, **batch)

    def configure_optimizers(self) -> dict[str, Any]:
        if self.train_args is None:
            raise ValueError("No TrainArgs provided")

        optimizer_type = (
            optim.AdamW
            if self.train_args.optimizer_type is None
            else self.train_args.optimizer_type
        )
        optimizer_kwargs = (
            {}
            if self.train_args.optimizer_kwargs is None
            else self.train_args.optimizer_kwargs
        )
        optimizer = optimizer_type(
            self._get_parameter_groups(), lr=self.train_args.lr, **optimizer_kwargs  # type: ignore
        )

        scheduler_type = (
            lr_scheduler.ReduceLROnPlateau
            if self.train_args.scheduler_type is None
            else self.train_args.scheduler_type
        )
        scheduler_kwargs = (
            {"mode": "min", "factor": 0.1, "patience": 5}
            if self.train_args.scheduler_kwargs is None
            else self.train_args.scheduler_kwargs
        )
        scheduler = scheduler_type(optimizer, **scheduler_kwargs)
        lr_scheduler_config = (
            {"monitor": "val_loss"}
            if self.train_args.lr_scheduler_config is None
            else self.train_args.lr_scheduler_config
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **lr_scheduler_config,
            },
        }

    def configure_callbacks(self) -> list[Callback]:
        if self.train_args is None:
            return []

        save_config = SaveConfigCallback(self.train_args.config_path)

        checkpoints_dir = os.path.join(
            self.train_args.checkpoints_location,
            self.model_name,
            self.train_args.project,
            self.train_args.run,
        )
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_last=True,
            dirpath=checkpoints_dir,
            filename="best_{epoch:02d}_{val_loss:0.4f}",
        )
        model_checkpoint.CHECKPOINT_NAME_LAST = "last_{epoch:02d}_{val_loss:0.4f}"  # type: ignore

        save_torch_weights = SaveTorchWeights(
            checkpoints_dir=checkpoints_dir,
            log=True,
            model_args=self.model_args,
            data_args=self.data_args,
            train_args=self.train_args,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-9,
            patience=11,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        callbacks = [
            save_config,
            model_checkpoint,
            save_torch_weights,
            early_stopping,
            lr_monitor,
        ]
        if self.train_args.callbacks is not None:
            callbacks += self.train_args.callbacks

        return callbacks

    # ================================ Train ================================

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(random.randint(1, 9999), verbose=False)

    def training_step(self, batch: SpecBatch) -> Tensor:
        train_loss = self._step("train", batch)
        return train_loss

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    # ================================ Validation ================================

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("val")

    def on_validation_epoch_start(self) -> None:
        pl.seed_everything(1234, verbose=False)

    def validation_step(self, batch: SpecBatch) -> None:
        self._step("val", batch)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    # ================================ Test ================================

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def on_test_epoch_start(self) -> None:
        pl.seed_everything(2345, verbose=False)

    def test_step(self, batch: SpecBatch) -> None:
        self._step("test", batch)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    # ================================ Predict ================================

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def on_predict_epoch_start(self) -> None:
        pl.seed_everything(2345, verbose=False)

    def predict_step(self, batch: SpecBatch) -> TrainModuleOut:
        batch = self._apply_batch_transform("test", batch)
        out = self(batch)

        return out
