"""Contrastive learning experiment parameters."""

import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "contrastive"
run = "bimamba-cont"

lr = 2e-4
epochs = 100
batch_size = 64
model_size = "tiny"
devices = [1, 2, 3, 4]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)
batch_transform = nn.Sequential(SpecBatchToContrastive(), SpecBatchContrastiveMask(104))


PARAMS = Params(
    train_module_type=ContModule,
    model_type=AudioMamba,
    model_args=AudioMambaArgs(
        spec_size=(128, 65),
        patch_size=(16, 5),
        channels=1,
        embed_dim=AUDIO_MAMBA_DEFAULT_CONFIG[model_size]["embed_dim"],
        depth=AUDIO_MAMBA_DEFAULT_CONFIG[model_size]["depth"],
        ssm_cfg=SSM_CONFIG,
        mask_ratio=0.0,
        mask_token_type="noise",
        head_drop_rate=0.3,
        cls_position="none",
        use_pred_head=True,
        use_rms_norm=True,
        fused_add_norm=True,
        bi_mamba_type="v1",
        output_type="mean",
    ),
    data_args=DataArgs(
        data_location=DATA_LOCATION,
        splits_dir=CONTRASTIVE_SPLITS_DIR,
        processor_type=SpecProcessor,
        batch_size=batch_size,
        num_workers=8,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        train_batch_transform=batch_transform,
        val_batch_transform=batch_transform,
        test_batch_transform=batch_transform,
    ),
    train_args=TrainArgs(
        config_path=__file__,
        loss=InfoNCELoss(temperature=0.01),
        lr=lr,
        max_epochs=epochs,
        project=project,
        run=run,
        checkpoints_location=CHECKPOINTS_LOCATION,
        scheduler_type=OneCycleLR,
        scheduler_kwargs={
            "max_lr": lr,
            "epochs": epochs,
            "steps_per_epoch": get_number_of_steps(
                CONTRASTIVE_SPLITS_DIR, batch_size, num_devices=len(devices)
            ),
            "pct_start": 0.3,
        },
        lr_scheduler_config={"interval": "step"},
        trainer_kwargs={
            "gradient_clip_val": 1.0,
        },
        devices=devices,
    ),
)
