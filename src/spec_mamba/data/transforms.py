"""Transforms."""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from spec_mamba.data.metadata import SpecBatch
from spec_mamba.data.utils import get_mask


class SpecTimeFirst(nn.Module):
    """(c, f, l) -> (l, c, f)"""

    def forward(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram.permute(-1, -3, -2)

        return spectrogram


class SpecChanFirst(nn.Module):
    """(l, c, f) -> (c, f, l)"""

    def forward(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram.permute(-2, -1, -3)

        return spectrogram


class SpecTranspose(nn.Module):
    """(c, f, l) <-> (c, l, f)"""

    def forward(self, spectrogram: Tensor) -> Tensor:
        spectrogram = spectrogram.transpose(-1, -2)

        return spectrogram


class SpecNormalize(nn.Module):
    """Clip spectrogram between [db_min, db_max] and normalize it linearly to [0, 1]."""

    def __init__(self, db_min: float, db_max: float) -> None:
        super().__init__()
        self.db_min = db_min
        self.db_max = db_max

    def forward(self, spectrogram: Tensor) -> Tensor:
        spectrogram = torch.clamp(spectrogram, self.db_min, self.db_max)
        spectrogram = (spectrogram - self.db_min) / (self.db_max - self.db_min)

        return spectrogram


class SpecBatchToContrastive(nn.Module):
    """
    Reshape a batch of 2 channel spectrograms by unpacking the 2 channels into
    the batch dimension for contrastive learning.

    Given a batch B of size (N, 2, H, W), the resulting batch B' has size
    (2N, 1, H, W) and satisfies for 0 <= i < N:
    - B'[i, 0, H, W] = B[i, 0, H, W]
    - B'[i + N, 0, H, W] = B[i, 1, H, W]
    """

    def forward(self, batch: SpecBatch) -> SpecBatch:
        if batch["spectrogram"].shape[1] != 2:
            raise ValueError(
                "Spectrograms must have 2 channels for contrastive reconstruction."
            )

        batch["sample_id"] = batch["sample_id"] + batch["sample_id"]
        batch["spectrogram"] = rearrange(batch["spectrogram"], "b c f t -> (c b) 1 f t")

        if ("target" in batch) and (batch["target"] is not None):
            batch["target"] = rearrange(batch["target"], "b c f t -> (c b) 1 f t")

        return batch


class SpecBatchContrastiveMask(nn.Module):
    """
    Apply a mask to the batch of unpacked spectrogram channels.
    Each channel is masked by 50% with the masks being complementary for the two channels.
    """

    def __init__(self, num_patches: int) -> None:
        super().__init__()
        self.num_patches = num_patches

    def forward(self, batch: SpecBatch) -> SpecBatch:
        if batch["spectrogram"].shape[0] % 2 != 0:
            raise ValueError("Batch size must be even for contrastive masking.")

        batch_size = batch["spectrogram"].shape[0] // 2
        mask_1 = get_mask(
            (batch_size, self.num_patches),
            mask_ratio=0.5,
            device=batch["spectrogram"].device,
        )
        mask_2 = 1 - mask_1
        batch["mask"] = torch.cat([mask_1, mask_2], dim=0).to(
            batch["spectrogram"].device
        )

        return batch


class SpecAddGaussianNoise(nn.Module):
    """Add random Gaussian noise to the spectrogram."""

    def __init__(self, mean: float = 0.0, std: float = 0.01) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, spectrogram: Tensor) -> Tensor:
        spectrogram += torch.randn_like(spectrogram) * self.std + self.mean
        spectrogram = spectrogram.type(torch.float32)

        return spectrogram
