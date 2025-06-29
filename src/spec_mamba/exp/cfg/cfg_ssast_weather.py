"""SSAST weather metadata experiment parameters."""

import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "weather"
run = "ssast-cont-1"

lr = 5e-4
epochs = 40
batch_size = 64
model_size = "tiny"
labels = ["precipRate_label", "windspeedAvg_label"]
devices = [4]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)


PARAMS = Params(
    train_module_type=CLFModule,
    model_type=SSASTCLF,
    model_args=SSASTCLFArgs(
        num_classes=2,
        spec_size=(128, 65),
        patch_size=(16, 5),
        channels=1,
        embed_dim=SSAST_DEFAULT_CONFIG[model_size]["embed_dim"],
        depth=SSAST_DEFAULT_CONFIG[model_size]["depth"],
        num_heads=SSAST_DEFAULT_CONFIG[model_size]["num_heads"],
        mlp_ratio=4,
        mask_ratio=0.0,
        mask_token_type="noise",
        clf_dropout=0.3,
        clf_hidden_features=SSAST_DEFAULT_CONFIG[model_size]["embed_dim"],
        cls_position="none",
        use_pred_head=False,
        use_rms_norm=True,
        output_type="mean",
    ),
    data_args=DataArgs(
        data_location=DATA_LOCATION,
        splits_dir=TARGETS_SPLITS_DIR,
        processor_type=SpecProcessor,
        labels=labels,
        labels_dtype=np.float32,
        batch_size=batch_size,
        num_workers=8,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    ),
    train_args=TrainArgs(
        config_path=__file__,
        loss=nn.HuberLoss(),
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
                TARGETS_SPLITS_DIR, batch_size=batch_size, num_devices=len(devices)
            ),
            "pct_start": 0.3,
        },
        lr_scheduler_config={"interval": "step"},
        checkpoint_path=get_checkpoint_path(
            CHECKPOINTS_LOCATION,
            "SSAST",
            "contrastive",
            "ssast-cont",
            mode="best",
            weights_only=True,
        ),
        freeze_pretrained=True,
        devices=devices,
    ),
)
