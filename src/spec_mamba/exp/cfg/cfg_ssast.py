"""SSAST experiment parameters."""

import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "foundation"
run = "ssast-all_data-tiny-MSE"

lr = 2e-4
epochs = 80
batch_size = 128
model_size = "tiny"
# devices = [4]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)

PARAMS = Params(
    train_module_type=RecModule,
    model_type=SSAST,
    model_args=SSASTArgs(
        spec_size=(128, 65),
        patch_size=(16, 5),
        channels=1,
        embed_dim=SSAST_DEFAULT_CONFIG[model_size]["embed_dim"],
        depth=SSAST_DEFAULT_CONFIG[model_size]["depth"],
        num_heads=SSAST_DEFAULT_CONFIG[model_size]["num_heads"],
        mlp_ratio=4,
        mask_ratio=0.5,
        cls_position="none",
        use_pred_head=True,
        use_rms_norm=True,
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
                FOUNDATION_SPLITS_DIR,
                batch_size=batch_size,
                # num_devices=len(devices),
            ),
            "pct_start": 0.3,
        },
        lr_scheduler_config={"interval": "step"},
        # devices=devices,
    ),
)
