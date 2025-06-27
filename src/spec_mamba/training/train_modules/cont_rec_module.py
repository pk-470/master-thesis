"""ContRecModule: PyTorch Lightning wrapper for training foundation models with contrastive reconstruction."""

from typing import Literal, Optional

from torch import Tensor
from torchmetrics import MetricCollection

from spec_mamba.models import SSAST, AudioMamba
from spec_mamba.training.args import AudioMambaArgs, SSASTArgs
from spec_mamba.training.losses import ContRecLoss
from spec_mamba.training.train_modules.train_module import SpecBatch, TrainModule


class ContRecModule(TrainModule):
    """PyTorch Lightning wrapper for training foundation models with contrastive reconstruction."""

    model_type: type[AudioMamba | SSAST]
    model_args: AudioMambaArgs | SSASTArgs
    model: AudioMamba | SSAST

    def _init_model(self) -> AudioMamba | SSAST:
        if not self.model_args.use_pred_head:
            raise ValueError("Set use_pred_head=True for contrastive reconstruction.")
        if self.model_args.channels != 1:
            raise ValueError("Set channels=1 for contrastive reconstruction.")
        if self.model_args.mask_ratio != 0.0:
            raise ValueError(
                "Masking should be handled by the SpecBatchContrastiveMask transform. "
                "Set mask_ratio=0.0 for contrastive learning."
            )
        if self.model_args.output_type not in ("full", "emb"):
            raise ValueError(
                "For contrastive reconstruction learning choose one of the expanded output types: 'full', 'emb'."
            )

        model = super()._init_model()
        assert isinstance(model, (AudioMamba, SSAST))

        return model

    def _init_loss(self) -> Optional[ContRecLoss]:
        if self.train_args is not None:
            if not isinstance(self.train_args.loss, ContRecLoss):
                raise ValueError("Use ContRecLoss for contrastive reconstruction.")
            return self.train_args.loss
        return None

    def _forward(
        self, mode: Literal["train", "val", "test"], batch: SpecBatch
    ) -> dict[str, Tensor]:
        if ("mask" not in batch) or (batch["mask"] is None):
            raise ValueError(
                "Use the SpecBatchContrastiveMask transform to add a mask for contrastive learning."
            )
        if ("target" not in batch) or (batch["target"] is None):
            raise ValueError("No target provided.")
        assert self.loss is not None

        x = batch["spectrogram"]
        target = batch["target"]
        mask = batch["mask"]

        logits, mask = self.model(x, mask=mask)

        if self.model.output_type == "full":
            cont = logits[:, 0, :]
            rec = logits[:, 1:, :]
        else:
            cont = logits.mean(dim=1)
            rec = logits

        rec, mask = self.model.reshape_as_spec(rec, mask)
        losses = self.loss(cont, rec, target, mask)

        metrics: Optional[MetricCollection] = getattr(self, f"{mode}_metrics")
        if metrics is not None:
            metrics.update(logits, target)

        return losses
