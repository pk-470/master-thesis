"""Tests for SSAST, SSASTCLF models."""

from typing import Literal

import pytest
import torch
from torch import Tensor

from spec_mamba.models.ssast import SSAST, SSASTCLF
from tests.conftest import train_one_epoch


@pytest.mark.parametrize(
    "use_pred_head, cls_position",
    [
        (True, "start"),
        (False, "middle"),
        (True, "end"),
        (False, "double"),
    ],
)
def test_ssast(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    use_pred_head: bool,
    cls_position: Literal["start", "middle", "end", "double"],
    device: str,
) -> None:

    model = (
        SSAST(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position=cls_position,
            output_type="full",
            use_pred_head=use_pred_head,
            mask_ratio=0.0,
            use_rms_norm=True,
        )
        .eval()
        .to(device)
    )

    last_dim = model.patch_dim if use_pred_head else embed_dim

    # Test full output
    with torch.no_grad():
        out_full, mask = model(sample)

    assert isinstance(out_full, Tensor) and isinstance(mask, Tensor)
    assert out_full.shape == (batch_size, model.num_patches + 1, last_dim)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Test cls output
    with torch.no_grad():
        out_cls, mask = model(sample, output_type="cls")

    assert isinstance(out_cls, Tensor) and isinstance(mask, Tensor)
    assert out_cls.shape == (batch_size, last_dim)
    assert torch.allclose(out_full[:, 0, :], out_cls)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Test embeddings output
    with torch.no_grad():
        out_emb, mask = model(sample, output_type="emb")

    assert isinstance(out_emb, Tensor) and isinstance(mask, Tensor)
    assert out_emb.shape == (batch_size, model.num_patches, last_dim)
    assert torch.allclose(out_full[:, 1:, :], out_emb)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Test last output
    with torch.no_grad():
        out_last, mask = model(sample, output_type="last")

    assert isinstance(out_last, Tensor) and isinstance(mask, Tensor)
    assert out_last.shape == (batch_size, last_dim)
    if not use_pred_head:
        assert torch.allclose(out_full[:, -1, :], out_last)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Test mean output
    with torch.no_grad():
        out_mean, mask = model(sample, output_type="mean")

    assert isinstance(out_mean, Tensor) and isinstance(mask, Tensor)
    assert out_mean.shape == (batch_size, last_dim)
    if not use_pred_head:
        assert torch.allclose(out_full[:, 1:, :].mean(dim=1), out_mean)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Test max output
    with torch.no_grad():
        out_max, mask = model(sample, output_type="max")

    assert isinstance(out_max, Tensor) and isinstance(mask, Tensor)
    assert out_max.shape == (batch_size, last_dim)
    if not use_pred_head:
        assert torch.allclose(out_full[:, 1:, :].max(dim=1).values, out_max)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() == 0.0

    # Free memory
    del model, out_full, out_cls, out_emb, out_last, out_mean, out_max, mask
    torch.cuda.empty_cache()


def test_ssast_train(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    device: str,
) -> None:
    model = SSAST(
        spec_size=spec_size,
        patch_size=patch_size,
        channels=channels,
        embed_dim=embed_dim,
        cls_position="none",
        output_type="emb",
        use_pred_head=True,
        mask_ratio=0.5,
    ).to(device)
    train_one_epoch(model, sample)


def test_ssast_clf(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    num_classes: int,
    device: str,
) -> None:

    model = (
        SSASTCLF(
            num_classes=num_classes,
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position="none",
            output_type="mean",
            use_pred_head=False,
            mask_ratio=0.0,
            use_rms_norm=False,
        )
        .eval()
        .to(device)
    )

    with torch.no_grad():
        logits = model(sample)

    assert isinstance(logits, Tensor)
    assert logits.shape == (batch_size, num_classes)

    # Free memory
    del model, logits
    torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
