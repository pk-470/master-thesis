"""RecModule: PyTorch Lightning wrapper for training foundation models with (masked) reconstruction."""

from typing import Literal, Optional

from torch import Tensor
from torchmetrics import MetricCollection

from spec_mamba.models import SSAST, AudioMamba
from spec_mamba.training.args import AudioMambaArgs, SSASTArgs
from spec_mamba.training.losses import MaskedLoss
from spec_mamba.training.train_modules.train_module import SpecBatch, TrainModule


class RecModule(TrainModule):
    """PyTorch Lightning wrapper for training foundation models with (masked) reconstruction."""

    model_type: type[AudioMamba | SSAST]
    model_args: AudioMambaArgs | SSASTArgs
    model: AudioMamba | SSAST

    def _init_model(self) -> AudioMamba | SSAST:
        if self.model_args.output_type != "emb":
            raise ValueError("Set output_type='emb' for reconstruction.")
        if not self.model_args.use_pred_head:
            raise ValueError("Set use_pred_head=True for reconstruction.")

        model = super()._init_model()
        assert isinstance(model, (AudioMamba, SSAST))

        return model

    def _forward(
        self, mode: Literal["train", "val", "test"], batch: SpecBatch
    ) -> dict[str, Tensor]:
        if ("target" not in batch) or (batch["target"] is None):
            raise ValueError("No target provided.")
        assert self.loss is not None

        x = batch["spectrogram"]
        target = batch["target"]

        logits, mask = self.model(x)
        logits, mask = self.model.reshape_as_spec(logits, mask)

        if isinstance(self.loss, MaskedLoss):
            loss = self.loss(logits, target, mask)
        else:
            loss = self.loss(logits, target)

        metrics: Optional[MetricCollection] = getattr(self, f"{mode}_metrics")
        if metrics is not None:
            metrics.update(logits, target)

        return {"loss": loss}
