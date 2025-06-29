"""AudioMamba experiment parameters."""

import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "foundation"
run = "bimamba-mse"

lr = 5e-4
epochs = 80
batch_size = 256
model_size = "tiny"
devices = [4]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)


PARAMS = Params(
    train_module_type=RecModule,
    model_type=AudioMamba,
    model_args=AudioMambaArgs(
        spec_size=(128, 65),
        patch_size=(16, 5),
        channels=1,
        embed_dim=AUDIO_MAMBA_DEFAULT_CONFIG[model_size]["embed_dim"],
        depth=AUDIO_MAMBA_DEFAULT_CONFIG[model_size]["depth"],
        ssm_cfg=SSM_CONFIG,
        mask_ratio=0.5,
        cls_position="none",
        use_pred_head=True,
        use_rms_norm=True,
        fused_add_norm=True,
        bi_mamba_type="v1",
        output_type="emb",
    ),
    data_args=DataArgs(
        data_location=DATA_LOCATION,
        splits_dir=FOUNDATION_SPLITS_DIR,
        processor_type=SpecProcessor,
        target_fn=nn.Identity(),
        batch_size=batch_size,
        num_workers=16,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    ),
    train_args=TrainArgs(
        config_path=__file__,
        loss=MaskedLoss(nn.MSELoss()),
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
                FOUNDATION_SPLITS_DIR, batch_size=batch_size, num_devices=len(devices)
            ),
            "pct_start": 0.3,
        },
        lr_scheduler_config={"interval": "step"},
        devices=devices,
    ),
)
