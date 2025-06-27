import os

import torch

from spec_mamba.training.args import *
from spec_mamba.training.callbacks import *
from spec_mamba.training.checkpoint import *
from spec_mamba.training.losses import *
from spec_mamba.training.train_modules import *

# Set CUDA device order to be the same as the one from nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Decrease precision for faster training
torch.set_float32_matmul_precision("high")
