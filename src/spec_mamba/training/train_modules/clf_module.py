"""CLFModule: PyTorch Lightning wrapper for training CLF models."""

from spec_mamba.models import SSASTCLF, AudioMambaCLF
from spec_mamba.training.args import AudioMambaCLFArgs, SSASTCLFArgs
from spec_mamba.training.checkpoint import (
    load_and_freeze_state_dict,
    load_backbone_from_checkpoint,
)
from spec_mamba.training.train_modules.train_module import TrainModule


class CLFModule(TrainModule):
    """PyTorch Lightning wrapper for training CLF models."""

    model_type: type[AudioMambaCLF | SSASTCLF]
    model_args: AudioMambaCLFArgs | SSASTCLFArgs
    model: AudioMambaCLF | SSASTCLF

    def _init_model(self) -> AudioMambaCLF | SSASTCLF:
        if self.model_args.mask_ratio > 0.0:
            raise ValueError("Set mask_ratio=0.0 for classifier training.")

        model = self.model_type(**self.model_args)

        if (self.train_args is not None) and (
            self.train_args.checkpoint_path is not None
        ):
            model = load_and_freeze_state_dict(
                model,
                self.train_args.checkpoint_path,
                strict=False,
                freeze_pretrained=self.train_args.freeze_pretrained,
                load_fn=load_backbone_from_checkpoint,
            )

        return model
