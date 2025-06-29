"""SSASTCLF experiment parameters."""

import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassPrecision,
    MulticlassRecall,
)

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "binary"
run = "ssast-cont"

lr = 1e-3
epochs = 100
batch_size = 64
model_size = "tiny"
label = "Bird_label"
devices = [2]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)

metrics = MetricCollection(
    {
        "accuracy": MulticlassAccuracy(num_classes=2, average="micro"),
        "precision": MulticlassPrecision(num_classes=2, average="none"),
        "recall": MulticlassRecall(num_classes=2, average="none"),
        "auroc": MulticlassAUROC(num_classes=2, average="none"),
        "average_precision": MulticlassAveragePrecision(num_classes=2, average="none"),
    }
)


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
        labels=label,
        labels_dtype=np.int64,
        batch_size=batch_size,
        num_workers=8,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
    ),
    train_args=TrainArgs(
        config_path=__file__,
        loss=nn.CrossEntropyLoss(weight=get_label_weights(TARGETS_SPLITS_DIR, label)),
        lr=lr,
        max_epochs=epochs,
        project=project,
        run=run,
        checkpoints_location=CHECKPOINTS_LOCATION,
        scheduler_type=ReduceLROnPlateau,
        scheduler_kwargs={
            "mode": "min",
            "factor": 0.1,
            "patience": 5,
            "threshold": 1e-6,
        },
        lr_scheduler_config={"monitor": "val_loss"},
        train_metrics=metrics,
        val_metrics=metrics,
        test_metrics=metrics,
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
