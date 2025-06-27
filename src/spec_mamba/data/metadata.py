"""Metadata for spectrograms and waveforms."""

from typing import Any, NotRequired, Optional, TypedDict

from torch import Tensor


class WavMetadata(TypedDict):
    """Waveform with metadata (output of WavProcessor)."""

    sample_id: str
    sample_rate: int
    waveform: Tensor


class SpecMetadata(TypedDict):
    """Spectrogram with metadata (output of WavToSpecProcessor and SpecProcessor)."""

    sample_id: str
    spectrogram: Tensor


class SpecSample(SpecMetadata):
    """Spectrogram training sample, including a target (output of GF23Dataset)."""

    target: NotRequired[Any | Tensor]


class SpecBatch(TypedDict):
    """Batch of spectrograms (dictionary used during training)."""

    sample_id: list[str]
    spectrogram: Tensor
    mask: NotRequired[Tensor]
    target: NotRequired[Optional[Tensor]]
