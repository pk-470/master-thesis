"""Common audio Mamba Block."""

from functools import partial
from typing import Any, Literal, Optional, TypeAlias

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.layers.drop import DropPath
from torch import Tensor

from bi_mamba_ssm.modules.mamba_simple import Mamba as BiMamba

MambaMixer: TypeAlias = type[Mamba] | partial[Mamba] | type[BiMamba] | partial[BiMamba]
NormLayer: TypeAlias = type[nn.LayerNorm | RMSNorm] | partial[nn.LayerNorm | RMSNorm]


class MambaBlock(nn.Module):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    """

    def __init__(
        self,
        dim: int,
        mixer_cls: MambaMixer,
        norm_cls: NormLayer = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_path: float = 0.0,
        layer_idx: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path)
        self.layer_idx = layer_idx
        if self.fused_add_norm and not isinstance(self.norm, (nn.LayerNorm, RMSNorm)):
            raise NotImplementedError(
                "Only LayerNorm and RMSNorm are supported for `fused_add_norm`."
            )

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
    ) -> tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            assert residual is not None
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:
                out = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                out = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            assert out is not None
            hidden_states, residual = out

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        assert residual is not None

        return hidden_states, residual

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype: Any = None, **kwargs
    ) -> tuple[Tensor, Tensor]:
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


def create_block(
    d_model: int,
    ssm_cfg: Optional[dict[str, Any]] = None,
    norm_epsilon: float = 1e-5,
    drop_path: float = 0.0,
    use_rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = False,
    layer_idx: Optional[int] = None,
    bi_mamba_type: Literal["none", "v1", "v2"] = "v1",
    init_layer_scale: Optional[int] = None,
    device: Any = None,
    dtype: Any = None,
) -> MambaBlock:
    """Create a MambaBlock."""
    factory_kwargs = {"device": device, "dtype": dtype}
    ssm_cfg = ssm_cfg if ssm_cfg is not None else {}

    if bi_mamba_type == "none":
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(
            BiMamba,
            layer_idx=layer_idx,
            bi_mamba_type=bi_mamba_type,
            init_layer_scale=init_layer_scale,
            **ssm_cfg,
            **factory_kwargs,
        )

    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs,
    )
    block = MambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        layer_idx=layer_idx,
    )

    return block
