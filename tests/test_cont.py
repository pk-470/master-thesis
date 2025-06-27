"""Tests for contrastive reconstruction."""

from math import log

import pytest
import torch
from torch import Tensor

from spec_mamba.data.transforms import SpecBatchContrastiveMask, SpecBatchToContrastive
from spec_mamba.training.losses import InfoNCELoss
from spec_mamba.training.train_modules.train_module import SpecBatch


def test_SpecBatchToContrastive(sample: Tensor) -> None:
    batch = SpecBatch(
        sample_id=["sample_1", "sample_2"],
        spectrogram=sample,
        target=sample,
    )
    reshaped_batch = SpecBatchToContrastive()(batch)

    assert reshaped_batch["sample_id"] == [
        "sample_1",
        "sample_2",
        "sample_1",
        "sample_2",
    ]
    assert torch.equal(
        reshaped_batch["spectrogram"],
        torch.stack(
            [
                sample[0, 0, :, :],
                sample[1, 0, :, :],
                sample[0, 1, :, :],
                sample[1, 1, :, :],
            ],
            dim=0,
        ).unsqueeze(1),
    )
    assert ("target" in reshaped_batch) and (reshaped_batch["target"] is not None)
    assert torch.equal(reshaped_batch["target"], reshaped_batch["spectrogram"])


def test_SpecBatchContrastiveMask(
    sample: Tensor, spec_size: tuple[int, int], patch_size: tuple[int, int]
) -> None:
    batch = SpecBatch(
        sample_id=["sample_1", "sample_2"],
        spectrogram=sample,
        target=None,
    )
    num_patches = spec_size[0] // patch_size[0] * spec_size[1] // patch_size[1]

    reshaped_batch = SpecBatchToContrastive()(batch)
    masked_batch = SpecBatchContrastiveMask(num_patches=num_patches)(reshaped_batch)

    assert ("mask" in masked_batch) and isinstance(masked_batch["mask"], Tensor)
    assert masked_batch["mask"].shape == (sample.shape[0] * 2, num_patches)
    assert masked_batch["mask"].sum() / masked_batch["mask"].numel() == 0.5
    assert torch.equal(masked_batch["mask"][:2, :], 1 - masked_batch["mask"][2:, :])


def test_info_nce_loss() -> None:
    x = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    loss_fn = InfoNCELoss(temperature=9e-15, reduction="none")
    loss = loss_fn(x)
    assert torch.allclose(loss, torch.tensor([0.0, -log(1 / 3), 0.0, -log(1 / 3)]))


if __name__ == "__main__":
    pytest.main([__file__])
