"""Contrastive reconstruction experiment parameters."""

import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from spec_mamba.data import *
from spec_mamba.exp.cfg.constants import *
from spec_mamba.exp.params import Params
from spec_mamba.exp.paths import *
from spec_mamba.models import *
from spec_mamba.training import *

project = "test-dev"
run = "test-cont-rec"

lr = 1.5e-4
epochs = 10
batch_size = 32
devices = [0]

transform = SpecNormalize(db_min=DB_MIN, db_max=DB_MAX)
batch_transform = nn.Sequential(SpecBatchToContrastive(), SpecBatchContrastiveMask(104))

PARAMS = Params(
    train_module_type=ContRecModule,
    model_type=AudioMamba,
    model_args=AudioMambaArgs(
        spec_size=(128, 65),
        patch_size=(16, 5),
        channels=1,
        embed_dim=128,
        depth=5,
        ssm_cfg=SSM_CONFIG,
        mask_ratio=0,
        cls_position="none",
        use_rms_norm=True,
        fused_add_norm=True,
        bi_mamba_type="v1",
        output_type="emb",
    ),
    data_args=DataArgs(
        data_location=WAVEFORMS_LOCATION,
        splits_dir=WAV_FOUNDATION_SPLITS_DIR,
        processor_type=WavToSpecProcessor,
        processor_kwargs={
            "add_channels": False,
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "mel_scale": True,
            "spec_kwargs": {"n_mels": N_MELS},
        },
        target_fn=nn.Identity(),
        batch_size=batch_size,
        num_workers=16,
        train_transform=transform,
        val_transform=transform,
        test_transform=transform,
        train_batch_transform=batch_transform,
        val_batch_transform=batch_transform,
        test_batch_transform=batch_transform,
    ),
    train_args=TrainArgs(
        config_path=__file__,
        loss=ContRecLoss(
            rec_loss=MaskedLoss(nn.L1Loss()),
            cont_loss=InfoNCELoss(temperature=0.1),
            cont_loss_weight=0.5,
        ),
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
                WAV_FOUNDATION_SPLITS_DIR,
                batch_size,
                num_devices=len(devices),
            ),
            "pct_start": 0.2,
        },
        lr_scheduler_config={"interval": "step"},
        devices=devices,
    ),
)
