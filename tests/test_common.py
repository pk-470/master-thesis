"""Tests for common preprocessing."""

from typing import Literal

import pytest
import torch
from torch import Tensor

from spec_mamba.models.audio_mamba import AudioMamba


def test_apply_masking(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    device: str,
) -> None:
    model = (
        AudioMamba(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position="none",
            output_type="emb",
            mask_ratio=0.8,
        )
        .eval()
        .to(device)
    )

    with torch.no_grad():
        x = model.patch_embed(sample, patch_size=patch_size, strides=patch_size)
        masked_x, mask = model.apply_masking(x)

    assert isinstance(x, Tensor)
    assert isinstance(model.mask_token, Tensor)
    assert model.mask_token.shape == (1, 1, embed_dim)
    assert masked_x.shape == x.shape == (batch_size, model.num_patches, embed_dim)
    assert mask.shape == (batch_size, model.num_patches)
    assert mask.sum() / mask.numel() == 0.8
    assert all(
        not torch.equal(mask[i, :], mask[j, :])
        for i in range(batch_size)
        for j in range(batch_size)
        if i != j
    )
    assert all(
        (
            torch.allclose(masked_x[i, j, :], model.mask_token.squeeze())
            if mask[i, j] == 1
            else torch.allclose(masked_x[i, j, :], x[i, j, :])
        )
        for i in range(batch_size)
        for j in range(model.num_patches)
    )


@pytest.mark.parametrize("cls_position", ("none", "start", "middle", "end", "double"))
def test_cls_token(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    cls_position: Literal["none", "start", "middle", "end", "double"],
    device: str,
) -> None:
    model = (
        AudioMamba(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position=cls_position,
            output_type="emb",
        )
        .eval()
        .to(device)
    )

    with torch.no_grad():
        x = model.patch_embed(sample, patch_size=patch_size, strides=patch_size)
        x_with_cls = model.add_cls_token(x)
        cls_token = model.get_cls_token(x_with_cls)
        x_without_cls = model.remove_cls_token(x_with_cls)

    assert isinstance(x, Tensor)

    if cls_position == "none":
        assert model.cls_token_position is None
        assert model.num_cls_tokens == 0
        assert model.cls_token is None
        assert cls_token is None

    else:
        assert model.cls_token is not None
        assert cls_token is not None

        if cls_position == "double":
            assert model.cls_token_position == (0, model.num_patches + 1)
            assert model.num_cls_tokens == 2
            assert x_with_cls.shape == (batch_size, model.num_patches + 2, embed_dim)
            assert model.cls_token.shape == (1, 2, embed_dim)
            assert cls_token.shape == (batch_size, 2, embed_dim)
            assert all(
                torch.allclose(x_with_cls[i, 0, :], model.cls_token[0, 0])
                for i in range(batch_size)
            )
            assert all(
                torch.allclose(x_with_cls[i, -1, :], model.cls_token[0, 1])
                for i in range(batch_size)
            )

        else:
            if cls_position == "start":
                assert model.cls_token_position == 0
            elif cls_position == "middle":
                assert model.cls_token_position == model.num_patches // 2
            elif cls_position == "end":
                assert model.cls_token_position == model.num_patches

            assert model.num_cls_tokens == 1
            assert x_with_cls.shape == (batch_size, model.num_patches + 1, embed_dim)
            assert model.cls_token.shape == (1, 1, embed_dim)
            assert cls_token.shape == (batch_size, embed_dim)
            assert all(
                torch.allclose(
                    x_with_cls[i, model.cls_token_position, :],
                    model.cls_token.squeeze(),
                )
                for i in range(batch_size)
            )

        assert all(
            torch.allclose(cls_token[i, ...].squeeze(), model.cls_token.squeeze())
            for i in range(batch_size)
        )

    assert x_without_cls.shape == x.shape
    assert torch.allclose(x_without_cls, x)


def test_transpose_token_sequence(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    device: str,
) -> None:
    model = (
        AudioMamba(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position="middle",
            output_type="emb",
        )
        .eval()
        .to(device)
    )

    with torch.no_grad():
        x = model.patch_embed(sample, patch_size=patch_size, strides=patch_size)
        x = model.add_cls_token(x)
        x_T = model.transpose_token_sequence(x)
        x_reshape = model.remove_cls_token(x).view(
            batch_size, *model.grid_size, embed_dim
        )
        x_T_reshape = model.remove_cls_token(x_T).view(
            batch_size, model.grid_size[1], model.grid_size[0], embed_dim
        )
        x_TT = model.transpose_token_sequence(x_T, transposed=True)

    assert x_T.shape == x.shape == (batch_size, model.num_patches + 1, embed_dim)
    assert torch.allclose(
        x_T[:, model.cls_token_position, :], x[:, model.cls_token_position, :]
    )
    assert torch.allclose(x_reshape, x_T_reshape.transpose(1, 2))
    assert torch.allclose(x_TT, x)


def test_reshape_as_spec(
    sample: Tensor,
    spec_size: tuple[int, int],
    patch_size: tuple[int, int],
    channels: int,
    embed_dim: int,
    batch_size: int,
    device: str,
) -> None:

    model = (
        AudioMamba(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            cls_position="none",
            output_type="emb",
            use_pred_head=True,
            mask_ratio=0.4,
        )
        .eval()
        .to(device)
    )

    with torch.no_grad():
        out_origin, mask_origin = model(sample)
        out_rec, mask_rec = model.reshape_as_spec(out_origin, mask_origin)

    assert isinstance(out_origin, Tensor) and isinstance(mask_origin, Tensor)
    assert torch.all(mask_origin.sum(dim=-1) / mask_origin.shape[-1] == 0.4)
    assert out_rec.shape == (batch_size, channels, *spec_size)
    assert mask_rec.shape == (batch_size, 1, *spec_size)
    assert all(
        torch.all(
            mask_rec[
                i,
                0,
                patch_size[0] * j : patch_size[0] * (j + 1),
                patch_size[1] * k : patch_size[1] * (k + 1),
            ]
            == mask_origin[i, (spec_size[1] // patch_size[1]) * j + k]
        )
        for i in range(batch_size)
        for j in range(spec_size[0] // patch_size[0])
        for k in range(spec_size[1] // patch_size[1])
    )


if __name__ == "__main__":
    pytest.main([__file__])
