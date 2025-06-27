"""SSAST model."""

from functools import partial
from typing import Any, Literal

import torch.nn as nn
from torch import Tensor

from spec_mamba.models.common import BaseCLF, BaseModel
from spec_mamba.models.ssast.layers import SSASTBlock


class SSAST(BaseModel):
    """SSAST model."""

    def __init__(
        self,
        spec_size: tuple[int, int] = (80, 200),
        patch_size: tuple[int, int] = (16, 4),
        channels: int = 1,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
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

        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        super().__init__(
            spec_size=spec_size,
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
            depth=depth,
            cls_position=cls_position,
            output_type=output_type,
            masking_mode=masking_mode,
            mask_token_type=mask_token_type,
            mask_ratio=mask_ratio,
            drop_path_rate=drop_path_rate,
            pos_drop_rate=pos_drop_rate,
            head_drop_rate=head_drop_rate,
            norm_epsilon=norm_epsilon,
            use_rms_norm=use_rms_norm,
            use_pred_head=use_pred_head,
            transpose=transpose,
            device=device,
            dtype=dtype,
        )

    def _init_blocks(self) -> None:
        self.blocks = nn.ModuleList(
            [
                SSASTBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    act_layer=nn.ReLU,
                    norm_layer=partial(
                        nn.LayerNorm if not self.use_rms_norm else nn.RMSNorm,
                        eps=self.norm_epsilon,
                        **self.factory_kwargs,
                    ),
                )
                for _ in range(self.depth)
            ]
        )

    def forward_blocks(self, x: Tensor, inference_params=None) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.enc_drop(x)
        x = self.norm_f(x)

        return x


class SSASTCLF(BaseCLF):
    """Trainable classifier for SSAST."""

    backbone_type = SSAST
    backbone: SSAST
