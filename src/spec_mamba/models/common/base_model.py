"""Base (abstract) model."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_
from torch import Tensor

from spec_mamba.data.utils import get_mask
from spec_mamba.models.common.tokenization import FlexiPatchEmbed, FlexiPosEmbed
from spec_mamba.models.common.utils import segm_init_weights


class BaseModel(nn.Module, ABC):
    """Base (abstract) model."""

    def __init__(
        self,
        spec_size: tuple[int, int] = (80, 200),
        patch_size: tuple[int, int] = (16, 4),
        channels: int = 1,
        embed_dim: int = 192,
        depth: int = 12,
        cls_position: Literal["none", "start", "middle", "end", "double"] = "middle",
        output_type: Literal["cls", "last", "mean", "max", "emb", "full"] = "cls",
        masking_mode: Literal["unstructured", "timestep"] = "unstructured",
        mask_token_type: Literal["learned", "noise", "zeros"] = "learned",
        mask_ratio: float = 0.5,
        drop_path_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        head_drop_rate: float = 0.0,
        norm_epsilon: float = 1e-5,
        use_rms_norm: bool = False,
        use_pred_head: bool = True,
        transpose: bool = False,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()

        if cls_position not in ("none", "start", "middle", "end", "double"):
            raise ValueError(
                "cls_position can be one of: 'none', 'start', 'middle', 'end', 'double'."
            )
        if (spec_size[0] % patch_size[0] != 0) or (spec_size[1] % patch_size[1] != 0):
            raise ValueError(
                "Spectrogram dimensions must be divisible by the patch size."
            )

        self._verify_output_type(output_type, cls_position)

        self.spec_size = spec_size
        self.strides = self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.cls_position = cls_position
        self.output_type = output_type
        self.mask_ratio = mask_ratio
        self.masking_mode = masking_mode
        self.mask_token_type = mask_token_type
        self.drop_path_rate = drop_path_rate
        self.pos_drop_rate = pos_drop_rate
        self.head_drop_rate = head_drop_rate
        self.norm_epsilon = norm_epsilon
        self.use_rms_norm = use_rms_norm
        self.use_pred_head = use_pred_head
        self.transpose = transpose
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self._init_mask_token()
        self._init_cls_tokens()
        self._init_blocks()
        self._init_heads()

        self.enc_drop = DropPath(drop_path_rate)

        self.patch_embed = FlexiPatchEmbed(
            patch_size=patch_size,
            strides=self.strides,
            in_chans=channels,
            embed_dim=embed_dim,
        )

        self.pos_embed = FlexiPosEmbed(
            input_size=spec_size,
            patch_size=patch_size,
            strides=self.strides,
            pos_grid_size=self.grid_size,
            embed_dim=embed_dim,
            n_prefix_tokens=self.num_cls_tokens,
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.norm_f = (nn.LayerNorm if not use_rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **self.factory_kwargs
        )

    @property
    def patch_dim(self) -> int:
        return self.patch_size[0] * self.patch_size[1] * self.channels

    @property
    def grid_size(self) -> tuple[int, int]:
        return (
            self.spec_size[0] // self.patch_size[0],
            self.spec_size[1] // self.patch_size[1],
        )

    @property
    def num_patches(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @property
    def total_num_patches(self) -> int:
        return self.num_patches + self.num_cls_tokens

    @property
    def no_weight_decay(self) -> tuple[str, ...]:
        return ("cls_token", "mask_token", "pos_embed")

    @staticmethod
    def _verify_output_type(output_type: Any, cls_position: str) -> None:
        if output_type not in ("cls", "last", "mean", "max", "emb", "full"):
            raise ValueError(
                "output_type can be one of: 'cls', 'last', 'mean', 'max', 'emb', 'full'."
            )
        if (output_type in ("cls", "full")) and (cls_position == "none"):
            raise ValueError("'cls', 'full' output types require a CLS token.")

    @abstractmethod
    def _init_blocks(self) -> None:
        self.blocks = nn.ModuleList()

    @abstractmethod
    def forward_blocks(self, x: Tensor, inference_params: Any = None) -> Tensor:
        pass

    def _init_mask_token(self) -> None:
        if self.mask_token_type == "learned":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.mask_token, std=0.02)
        else:
            self.mask_token = None

    def _init_cls_tokens(self) -> None:
        self.cls_token = None
        self.num_cls_tokens = 0
        self.cls_token_position = None

        if self.cls_position == "double":
            self.cls_token = nn.Parameter(torch.zeros(1, 2, self.embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
            self.num_cls_tokens = 2
            self.cls_token_position = (0, self.num_patches + 1)

        elif self.cls_position != "none":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
            self.num_cls_tokens = 1
            if self.cls_position == "start":
                self.cls_token_position = 0
            if self.cls_position == "middle":
                self.cls_token_position = self.num_patches // 2
            if self.cls_position == "end":
                self.cls_token_position = self.num_patches

    def _init_heads(self) -> None:
        self.cls_head: Optional[nn.Module] = None
        self.head: Optional[nn.Module] = None

        if self.use_pred_head:
            if self.output_type == "full":
                self.cls_head = nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.ReLU(),
                    nn.Dropout(self.head_drop_rate),
                    nn.Linear(self.embed_dim, self.patch_dim),
                )
                self.cls_head.apply(segm_init_weights)
            self.head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(self.head_drop_rate),
                nn.Linear(self.embed_dim, self.patch_dim),
            )
            self.head.apply(segm_init_weights)

    def add_cls_token(self, x: Tensor, cls_token: Optional[Tensor] = None) -> Tensor:
        if self.cls_position == "none":
            return x

        B = x.shape[0]

        if cls_token is None:
            assert self.cls_token is not None
            cls_token = self.cls_token.expand(B, -1, -1)

        if self.cls_position == "double":
            assert cls_token.shape == (B, 2, self.embed_dim)
            x = torch.cat(
                (cls_token[:, 0, :].unsqueeze(1), x, cls_token[:, 1, :].unsqueeze(1)),
                dim=1,
            )

        else:
            cls_token = cls_token.view(B, 1, self.embed_dim)
            x = torch.cat(
                (
                    x[:, : self.cls_token_position, :],
                    cls_token,
                    x[:, self.cls_token_position :, :],
                ),
                dim=1,
            )

        return x

    def get_cls_token(self, x: Tensor) -> Optional[Tensor]:
        if self.cls_position == "double":
            return torch.cat([x[:, 0, :].unsqueeze(1), x[:, -1, :].unsqueeze(1)], dim=1)
        if self.cls_position != "none":
            return x[:, self.cls_token_position, :]
        return None

    def remove_cls_token(self, x: Tensor) -> Tensor:
        if self.cls_position == "double":
            x = x[:, 1:-1, :]
        elif self.cls_position != "none":
            assert isinstance(self.cls_token_position, int)
            x = torch.cat(
                (
                    x[:, : self.cls_token_position, :],
                    x[:, self.cls_token_position + 1 :, :],
                ),
                dim=1,
            )

        return x

    def transpose_token_sequence(self, x: Tensor, transposed=False) -> Tensor:
        B = x.shape[0]
        _F, _T = (
            self.spec_size[0] // self.patch_size[0],
            self.spec_size[1] // self.patch_size[1],
        )
        H, W = (_T, _F) if transposed else (_F, _T)

        cls_token = self.get_cls_token(x)
        x = self.remove_cls_token(x)

        x = x.reshape(B, H, W, -1)
        x = x.transpose(1, 2)
        x = x.reshape(B, W * H, -1)

        x = self.add_cls_token(x, cls_token=cls_token)

        return x

    def apply_masking(
        self,
        x: Tensor,
        mask_ratio: Optional[float] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if (mask is not None) and (mask_ratio is not None):
            raise ValueError(
                "'mask' and 'mask_ratio' cannot be provided at the same time."
            )

        if mask is None:
            mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio  # type: ignore
            if mask_ratio == 0:
                return x, torch.zeros(x.shape[:-1], device=x.device)

            mask = get_mask(
                x.shape[:-1],
                mask_ratio=mask_ratio,
                device=x.device,
                grid_width=self.grid_size[1],
                masking_mode=self.masking_mode,  # type: ignore
            )

        if self.mask_token_type == "learned":
            assert self.mask_token is not None
            mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1], 1)
        elif self.mask_token_type == "noise":
            mask_tokens = torch.empty_like(x)
            trunc_normal_(mask_tokens, std=0.02)
        else:
            mask_tokens = torch.zeros_like(x)

        x = x * (1 - mask.unsqueeze(-1)) + mask_tokens * mask.unsqueeze(-1)

        return x, mask

    def reshape_as_spec(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x = x.view(x.shape[0], self.channels, *self.spec_size)
        mask = (
            mask.view(mask.shape[0], 1, *self.grid_size)
            .repeat_interleave(self.patch_size[0], dim=-2)
            .repeat_interleave(self.patch_size[1], dim=-1)
        )

        return x, mask

    def forward_features(
        self,
        x: Tensor,
        output_type: Optional[
            Literal["cls", "last", "mean", "max", "emb", "full"]
        ] = None,
        mask_ratio: Optional[float] = None,
        mask: Optional[Tensor] = None,
        inference_params: Any = None,
    ) -> tuple[Tensor, Tensor]:

        output_type = output_type if output_type is not None else self.output_type  # type: ignore
        self._verify_output_type(output_type, self.cls_position)

        # 1) Patch embedding
        x = self.patch_embed(x, patch_size=self.patch_size, strides=self.strides)

        # 2) Masking
        x, mask = self.apply_masking(x, mask_ratio=mask_ratio, mask=mask)

        # 3) Add CLS token
        x = self.add_cls_token(x)

        # 4) Add positional embedding
        x = self.pos_embed(
            x,
            token_position=self.cls_token_position,
            patch_size=self.patch_size,
            strides=self.strides,
        )
        x = self.pos_drop(x)

        # 5) Forward pass
        if self.transpose:
            x = self.transpose_token_sequence(x)
        hidden_states = self.forward_blocks(x, inference_params=inference_params)
        if self.transpose:
            hidden_states = self.transpose_token_sequence(
                hidden_states, transposed=True
            )

        cls_token = self.get_cls_token(hidden_states)
        if self.cls_position == "double":
            assert cls_token is not None
            cls_token = cls_token.mean(dim=1)

        if output_type == "cls":
            assert cls_token is not None
            return cls_token, mask

        hidden_states = self.remove_cls_token(hidden_states)

        if output_type == "full":
            assert cls_token is not None
            out = torch.cat((cls_token.unsqueeze(1), hidden_states), dim=1)
        elif output_type == "last":
            out = hidden_states[:, -1, :]
        elif output_type == "mean":
            out = hidden_states.mean(dim=1)
        elif output_type == "max":
            out = hidden_states.max(dim=1).values
        else:
            out = hidden_states

        return out, mask

    def project(
        self,
        x: Tensor,
        output_type: Optional[
            Literal["cls", "last", "mean", "max", "emb", "full"]
        ] = None,
    ) -> Tensor:
        output_type = output_type if output_type is not None else self.output_type  # type: ignore
        self._verify_output_type(output_type, self.cls_position)

        if output_type == "full":
            if self.cls_head is not None and self.head is not None:
                cls_out: Tensor = self.cls_head(x[:, 0, :])
                emb_out: Tensor = self.head(x[:, 1:, :])
                out = torch.cat((cls_out.unsqueeze(1), emb_out), dim=1)
            else:
                out = x
        elif (output_type == "cls") and (self.cls_head is not None):
            out = self.cls_head(x)
        elif self.head is not None:
            out = self.head(x)
        else:
            out = x

        return out

    def forward(
        self,
        x: Tensor,
        output_type: Optional[
            Literal["cls", "last", "mean", "max", "emb", "full"]
        ] = None,
        mask_ratio: Optional[float] = None,
        mask: Optional[Tensor] = None,
        inference_params: Any = None,
    ) -> tuple[Tensor, Tensor]:

        output_type = output_type if output_type is not None else self.output_type  # type: ignore
        x, mask = self.forward_features(
            x,
            output_type=output_type,
            mask_ratio=mask_ratio,
            mask=mask,
            inference_params=inference_params,
        )
        out = self.project(x, output_type=output_type)

        return out, mask
