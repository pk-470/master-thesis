"""AudioMamba model and classifier."""

from functools import partial
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from torch import Tensor

from spec_mamba.models.audio_mamba.mamba_block import create_block
from spec_mamba.models.common import BaseCLF, BaseModel, init_weights


class AudioMamba(BaseModel):
    """AudioMamba model."""

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
        bi_mamba_type: Literal["none", "v1", "v2"] = "v1",
        ssm_cfg: Optional[dict[str, Any]] = None,
        initializer_cfg: Optional[dict[str, Any]] = None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        init_layer_scale: Optional[int] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:

        if bi_mamba_type not in ("none", "v1", "v2"):
            raise ValueError("bi_mamba_type can be one of: 'none', 'v1', 'v2'.")

        self.bi_mamba_type = bi_mamba_type
        self.ssm_cfg = ssm_cfg if ssm_cfg is not None else {}
        self.initializer_cfg = initializer_cfg if initializer_cfg is not None else {}
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.init_layer_scale = init_layer_scale

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
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr

        self.blocks = nn.ModuleList(
            [
                create_block(
                    self.embed_dim,
                    ssm_cfg=self.ssm_cfg,
                    norm_epsilon=self.norm_epsilon,
                    use_rms_norm=self.use_rms_norm,
                    residual_in_fp32=self.residual_in_fp32,
                    fused_add_norm=self.fused_add_norm,
                    layer_idx=i,
                    bi_mamba_type=self.bi_mamba_type,  # type: ignore
                    drop_path=inter_dpr[i],
                    init_layer_scale=self.init_layer_scale,
                    **self.factory_kwargs,
                )
                for i in range(self.depth)
            ]
        )
        self.apply(
            partial(
                init_weights,
                n_layer=self.depth,
                **self.initializer_cfg,
            )
        )

    def forward_blocks(
        self,
        x: Tensor,
        inference_params: Any = None,
    ) -> Tensor:

        residual = None
        hidden_states = x

        for block in self.blocks:
            hidden_states, residual = block(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.enc_drop(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.enc_drop(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            assert hidden_states is not None

        return hidden_states

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: Any = None, **kwargs
    ) -> dict[int, tuple[Tensor, Tensor]]:
        return {
            i: block.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
            for i, block in enumerate(self.blocks)
        }


class AudioMambaCLF(BaseCLF):
    """Trainable classifier for AudioMamba."""

    backbone_type = AudioMamba
    backbone: AudioMamba
