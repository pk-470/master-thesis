"""ContModule: PyTorch Lightning wrapper for training foundation models with contrastive learning."""

from typing import Literal, Optional

from torch import Tensor
from torchmetrics import MetricCollection

from spec_mamba.models import SSAST, AudioMamba
from spec_mamba.training.args import AudioMambaArgs, SSASTArgs
from spec_mamba.training.train_modules.train_module import SpecBatch, TrainModule


class ContModule(TrainModule):
    """PyTorch Lightning wrapper for training foundation models with contrastive learning."""

    model_type: type[AudioMamba | SSAST]
    model_args: AudioMambaArgs | SSASTArgs
    model: AudioMamba | SSAST

    def _init_model(self) -> AudioMamba | SSAST:
        if not self.model_args.use_pred_head:
            raise ValueError("Set use_pred_head=True for contrastive learning.")
        if self.model_args.channels != 1:
            raise ValueError("Set channels=1 for contrastive learning.")
        if self.model_args.mask_ratio != 0.0:
            raise ValueError(
                "Masking should be handled by the SpecBatchContrastiveMask transform. "
                "Set mask_ratio=0.0 for contrastive learning."
            )
        if self.model_args.output_type not in ("cls", "last", "mean", "max"):
            raise ValueError(
                "For contrastive learning choose one of the reduced output types: 'cls', 'last', 'mean', 'max'."
            )

        model = super()._init_model()
        assert isinstance(model, (AudioMamba, SSAST))

        return model

    def _forward(
        self, mode: Literal["train", "val", "test"], batch: SpecBatch
    ) -> dict[str, Tensor]:
        if ("mask" not in batch) or (batch["mask"] is None):
            raise ValueError(
                "Use the SpecBatchContrastiveMask transform to add a mask for contrastive learning."
            )
        assert self.train_args is not None
        assert self.loss is not None

        x = batch["spectrogram"]
        mask = batch["mask"]
        B = x.size(0) // 2

        logits, _ = self.model(x, mask=mask)

        if self.trainer.world_size > 1:
            logits = self.all_gather(logits, sync_grads=True)
            assert isinstance(logits, Tensor) and logits.dim() == 3
            W, _, D = logits.shape
            g = logits.view(W, 2, B, D)
            g = g.permute(1, 0, 2, 3).contiguous()  # (2, W, B, D)
            logits = g.view(2 * W * B, D)  # (2WB, D)

        loss = self.loss(logits)

        metrics: Optional[MetricCollection] = getattr(self, f"{mode}_metrics")
        if metrics is not None:
            z1, z2 = logits.chunk(2, dim=0)
            metrics.update(z1, z2)

        return {"loss": loss}
