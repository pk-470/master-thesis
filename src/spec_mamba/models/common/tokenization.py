# From FlexiAST: https://arxiv.org/pdf/2307.09286.pdf

import math
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.weight_init import lecun_normal_, trunc_normal_
from torch import Tensor

from spec_mamba.models.common.utils import to_2tuple


def divs(n: int) -> list[int]:
    return [i for i in range(1, n + 1) if n % i == 0]


def gcd(a: int, b: int) -> int:
    if b == 0:
        return a
    return gcd(b, a % b)


def resample_abs_pos_embed(
    posemb: Tensor,
    new_size: Sequence[int],
    old_size: Optional[Sequence[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bilinear",
    antialias: bool = True,  # Google uses True (implicitly)
    verbose: bool = False,
    pos_embed_prefix: bool = True,
) -> Tensor:
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = to_2tuple(int(math.sqrt(posemb.shape[1] - num_prefix_tokens)))
    else:
        old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens > 0 and pos_embed_prefix:  # TODO: CHECK THIS!!!
        posemb_prefix, posemb = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix = None

    # do the interpolation
    assert old_size is not None
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(
        posemb, size=new_size, mode=interpolation, antialias=antialias
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if verbose:
        # TODO: Implement logging here
        # _logger.info(f'Resized position embedding: {old_size} to {new_size}.')
        pass

    # add back extra (class, etc) prefix tokens
    if num_prefix_tokens > 0 and pos_embed_prefix:
        assert posemb_prefix is not None
        if verbose:
            print(posemb_prefix.shape, posemb.shape)
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def get_resize_mat_pinv(
    old_size: list[int],
    new_size: list[int],
    interpolation: str = "bilinear",
    antialias: bool = False,
) -> Tensor:

    assert len(old_size) == 2, "Old shape should only be hw"
    assert len(new_size) == 2, "New shape should only be hw"

    if tuple(old_size) == tuple(new_size):
        return torch.eye(np.prod(old_size))  # type: ignore

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias
        )[0, 0, ...].numpy()

        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.0
            mat.append(resize(basis_vec, _new_size).reshape(-1))

        return np.stack(mat).T

    resize_mat = get_resize_mat(
        old_size, new_size
    )  # This might be the B mentioned in the paper.

    try:
        resize_mat_pinv = torch.Tensor(np.linalg.pinv(resize_mat.T))
    except:
        resize_mat_pinv = torch.linalg.pinv(torch.Tensor(resize_mat.T))

    return resize_mat_pinv


def resample_patch_embed(
    patch_embed: Tensor,
    new_size: list[int],
    interpolation: str = "bilinear",
    antialias: bool = False,
    resize_mat_pinv=None,
) -> Tensor:
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """

    old_size = patch_embed.shape[-2:]

    if old_size == new_size:
        return patch_embed

    if resize_mat_pinv is None:
        resize_mat_pinv = get_resize_mat_pinv(
            old_size=old_size,  # type: ignore
            new_size=new_size,
            interpolation=interpolation,
            antialias=antialias,
        ).detach()

    # new^2 old^w,768 1 old^2 -> 768 1 new^2
    ens = torch.einsum(
        "xk,abk->abx",
        [
            resize_mat_pinv.to(patch_embed.device),
            patch_embed.reshape(patch_embed.size(0), patch_embed.size(1), -1),
        ],
    ).reshape(patch_embed.size(0), patch_embed.size(1), new_size[0], new_size[1])

    return ens


def vanilla_resample_patch_embed(
    patch_embed: Tensor,
    new_size: Sequence[int],
    interpolation: str = "bilinear",
    antialias: bool = True,  # Google uses True (implicitly)
) -> Tensor:

    B, C, H, W = patch_embed.shape

    new_size = to_2tuple(new_size)
    old_size = to_2tuple((H, W))
    if new_size == old_size:  # might not both be same container type
        return patch_embed

    # do the interpolation
    patch_embed = F.interpolate(
        patch_embed, size=new_size, mode=interpolation, antialias=antialias
    )

    return patch_embed


def get_shape(
    fstride: int,
    tstride: int,
    patch_size: int,
    input_fdim: int = 128,
    input_tdim: int = 1024,
) -> tuple[int, int]:
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(
        1, 768, kernel_size=(patch_size, patch_size), stride=(fstride, tstride)
    )
    test_out: Tensor = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]

    return f_dim, t_dim


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int] = (16, 16),
        strides: tuple[int, int] = (16, 16),
        in_chans: int = 1,
        embed_dim: int = 192,
        bias: bool = True,
        norm_layer: Any = None,
        flatten: bool = True,
        proj_load: Optional[nn.Conv2d] = None,
        resize_func=resample_patch_embed,
        precompute_for: Optional[list] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if verbose:
            print(f"Resize function is {resize_func.__name__}")
            print("Initializing FlexiPatchEmbed with the following parameters:")
            print(
                f'patch_size={patch_size}, in_chans={in_chans}, embed_dim={embed_dim}, bias={bias}, norm_layer={norm_layer}, flatten={flatten}, proj_load={"yes" if proj_load is not None else None}, resize_func={resize_func.__name__}'
            )

        self.patch_size: tuple[int, int] = to_2tuple(patch_size)
        self.strides: tuple[int, int] = to_2tuple(strides)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=strides, bias=bias
        )
        self.resize_func = resize_func

        lecun_normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        if verbose:
            print(f"The resize function is {self.resize_func.__name__}")

        if proj_load is not None:
            assert self.proj.bias is not None and proj_load.bias is not None

            if verbose:
                print("Loading projection weights!")
                print(
                    f"The shapes of the current projection: bias={self.proj.bias.shape}, weight={self.proj.weight.shape}"
                )
                print(
                    f"The shapes of the loaded projection: bias={proj_load.bias.shape}, weight={proj_load.weight.shape}"
                )

            if proj_load.bias.shape != self.proj.bias.shape:
                raise ValueError(
                    "The bias shape of the loaded projection layer does not match the current projection layer"
                )

            # copy the bias
            self.proj.bias = nn.Parameter(proj_load.bias)

            if proj_load.weight.shape != self.proj.weight.shape:
                self.proj.weight = nn.Parameter(
                    self.resize_func(proj_load.weight, list(self.patch_size))
                )
                if verbose:
                    print(
                        f"Resized the projection weights with {self.resize_func.__name__}"
                    )
                    print(
                        f"The shapes of the resized projection weights={self.proj.weight.shape}"
                    )
            else:
                self.proj.weight = nn.Parameter(proj_load.weight)

        self.precomputed_matrices = {}

        if precompute_for is not None:
            if not isinstance(precompute_for, list):
                raise ValueError("The precompute_for should be either None or a list!")

            if self.resize_func.__name__ != resample_patch_embed.__name__:
                raise ValueError(
                    "The precompute_for is only supported when the resize_func is resample_patch_embed!"
                )

            precompute_for = [
                to_2tuple(patch_size) for patch_size in list(precompute_for)
            ]

            for patch_size in precompute_for:
                self.precomputed_matrices[patch_size] = get_resize_mat_pinv(
                    list(self.patch_size),
                    list(patch_size),
                ).detach()

            if verbose:
                print(f"Precomputed weights for {precompute_for}")

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[tuple[int, int]] = None,
        strides: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        B, C, H, W = x.shape

        if patch_size is None:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        if strides is None:
            strides = self.strides
        strides = to_2tuple(strides)

        assert patch_size is not None and strides is not None

        if patch_size == self.patch_size:
            weight = self.proj.weight
        elif patch_size in self.precomputed_matrices:
            weight = self.resize_func(
                self.proj.weight,
                list(patch_size),
                resize_mat_pinv=self.precomputed_matrices[patch_size].to(x.device),
            )
        else:
            weight = self.resize_func(self.proj.weight, list(patch_size))

        bias = self.proj.bias

        x = F.conv2d(x, weight, bias=bias, stride=strides)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)

        return x


class FlexiPosEmbed(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int] = (128, 1024),
        patch_size: tuple[int, int] = (16, 16),
        strides: tuple[int, int] = (16, 16),
        pos_grid_size: Optional[tuple[int, int]] = (8, 64),
        embed_dim: int = 192,
        pos_embed_load: Optional[Tensor] = None,
        pos_grid_size_load: Optional[tuple[int, int]] = None,
        n_prefix_tokens: int = 1,  # Assuming there is a cls token by default
        pos_embed_prefix: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if verbose:
            print("Initializing FlexiPosEmbed with the following parameters:")
            print(
                f"input_size={input_size}, pos_grid_size={pos_grid_size}, embed_dim={embed_dim}, pos_embed_load={pos_embed_load.shape if pos_embed_load is not None else None}, pos_grid_size_load={pos_grid_size_load}, n_prefix_tokens={n_prefix_tokens}, pos_embed_prefix={pos_embed_prefix}"
            )

        self.input_size: tuple[int, int] = to_2tuple(input_size)
        self.strides: tuple[int, int] = to_2tuple(strides)
        self.patch_size: tuple[int, int] = to_2tuple(patch_size)
        self.pos_grid_size: tuple[int, int]

        if pos_grid_size is None:
            self.pos_grid_size = to_2tuple(
                FlexiPosEmbed.get_shape(*strides, patch_size, *input_size)
            )
        else:
            self.pos_grid_size = to_2tuple(pos_grid_size)

        pos_grid_size_load = (
            to_2tuple(pos_grid_size_load) if pos_grid_size_load is not None else None
        )

        num_patches = self.pos_grid_size[0] * self.pos_grid_size[1]
        self.n_prefix_tokens = n_prefix_tokens
        self.pos_embed_prefix = pos_embed_prefix
        self.embed_dim = embed_dim
        pos_embed_shape = (
            1,
            num_patches + (n_prefix_tokens if pos_embed_prefix else 0),
            embed_dim,
        )

        if pos_embed_load is not None:

            if verbose:
                print("Loading position embedding!")
                print(f"The shape of the current grid size: {pos_grid_size}")
                print(f"The shape of the loaded grid size: {pos_grid_size_load}")

            if pos_grid_size_load is None:
                raise ValueError(
                    "The loaded position embedding does not have the grid size information"
                )

            if pos_grid_size != pos_grid_size_load:
                self.pos_embed = nn.Parameter(
                    resample_abs_pos_embed(
                        pos_embed_load,
                        new_size=self.pos_grid_size,
                        old_size=pos_grid_size_load,
                        num_prefix_tokens=n_prefix_tokens,
                        pos_embed_prefix=self.pos_embed_prefix,
                    )
                )
                if verbose:
                    print(
                        f"The shape of the resampled position embedding: {self.pos_embed.shape}"
                    )
            else:
                self.pos_embed = nn.Parameter(pos_embed_load)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(*pos_embed_shape))
            self.pos_embed = trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def get_shape(
        fstride: int,
        tstride: int,
        patch_size: int | tuple[int, int],
        input_fdim: int,
        input_tdim: int,
    ) -> tuple[int, int]:
        patch_size = to_2tuple(patch_size)
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, 1, kernel_size=patch_size, stride=(fstride, tstride))
        test_out: Tensor = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]

        return f_dim, t_dim

    @staticmethod
    def insert_to_prefix(x: Tensor, from_poses: int | Sequence[int]) -> Tensor:
        if isinstance(from_poses, int):
            from_poses = [from_poses]
        for i, from_pos in enumerate(from_poses):
            x = torch.cat(
                [
                    x[:, :i],
                    x[:, from_pos : from_pos + 1],
                    x[:, i:from_pos],
                    x[:, from_pos + 1 :],
                ],
                dim=1,
            )

        return x

    @staticmethod
    def insert_from_prefix(x: Tensor, to_poses: int | Sequence[int]) -> Tensor:
        if isinstance(to_poses, int):
            to_poses = [to_poses]
        x_prefix, x = x[:, : len(to_poses)], x[:, len(to_poses) :]
        for i, to_pos in enumerate(to_poses):
            x = torch.cat([x[:, :to_pos], x_prefix[:, i : i + 1], x[:, to_pos:]], dim=1)

        return x

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[tuple[int, int]] = None,
        strides: Optional[tuple[int, int]] = None,
        token_position: Optional[int | tuple[int, int]] = None,
    ) -> Tensor:

        if token_position is not None:
            x = FlexiPosEmbed.insert_to_prefix(x, from_poses=token_position)

        if patch_size is None and strides is None:
            x = x + self.pos_embed
        else:
            if patch_size is None:
                patch_size = self.patch_size
            patch_size = to_2tuple(patch_size)

            if strides is None:
                strides = self.strides
            strides = to_2tuple(strides)

            assert patch_size is not None and strides is not None

            old_size = self.pos_grid_size
            new_size: list[int] = [
                *FlexiPosEmbed.get_shape(*strides, patch_size, *self.input_size)
            ]
            forward_pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=new_size,
                old_size=old_size,
                num_prefix_tokens=self.n_prefix_tokens,
                pos_embed_prefix=self.pos_embed_prefix,
            )

            if not self.pos_embed_prefix:
                final_patches = (
                    x[:, : self.n_prefix_tokens],
                    (x[:, self.n_prefix_tokens :] + forward_pos_embed),
                )
                x = torch.cat(final_patches, dim=1)
            else:
                x = x + forward_pos_embed

        if token_position is not None:
            x = FlexiPosEmbed.insert_from_prefix(x, to_poses=token_position)

        return x
