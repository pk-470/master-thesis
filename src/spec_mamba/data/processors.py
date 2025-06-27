"""Processor classes."""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from torchvision.transforms import Compose

from spec_mamba.data.metadata import SpecMetadata, WavMetadata
from spec_mamba.data.utils import clean_dataframe, compute_spectrogram


class BaseProcessor(ABC):
    """Base processor."""

    def __init__(
        self,
        location: str,
        dataframe: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.location = location
        self.dataframe = clean_dataframe(dataframe)

    @abstractmethod
    def process(self, idx: int) -> SpecMetadata | WavMetadata:
        pass


class WavProcessor(BaseProcessor):
    """Processes a single waveform, returns WavMetadata."""

    def __init__(
        self,
        location: str,
        dataframe: pd.DataFrame,
        sample_rate: Optional[int] = None,
        add_channels: bool = True,
        wav_transform: Optional[nn.Module | Compose] = None,
    ) -> None:
        super().__init__(location=location, dataframe=dataframe)
        self.sample_rate = sample_rate
        self.add_channels = add_channels
        self.wav_transform = wav_transform

    def _load_waveform(self, idx: int) -> tuple[torch.Tensor, int, pd.Series]:
        item = self.dataframe.iloc[idx]
        waveform, sample_rate = torchaudio.load(
            os.path.join(self.location, item["filename"])
        )

        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = F.resample(waveform, sample_rate, self.sample_rate)
            sample_rate = self.sample_rate

        if self.add_channels:
            waveform = torch.mean(waveform, dim=0).unsqueeze(0)  # (1, t)

        return waveform, sample_rate, item

    def process(self, idx: int) -> WavMetadata:
        waveform, sample_rate, item = self._load_waveform(idx)

        if self.wav_transform is not None:
            waveform = self.wav_transform(waveform)

        return WavMetadata(
            sample_id=item["time"], sample_rate=sample_rate, waveform=waveform
        )


class WavToSpecProcessor(WavProcessor):
    """Processes a single waveform, returns SpecMetadata."""

    def __init__(
        self,
        location: str,
        dataframe: pd.DataFrame,
        sample_rate: Optional[int] = None,
        add_channels: bool = True,
        n_fft: int = 1024,
        mel_scale: bool = True,
        spec_kwargs: Optional[dict[str, Any]] = None,
        wav_transform: Optional[nn.Module | Compose] = None,
        spec_transform: Optional[nn.Module | Compose] = None,
    ) -> None:
        super().__init__(
            location=location,
            dataframe=dataframe,
            sample_rate=sample_rate,
            add_channels=add_channels,
            wav_transform=wav_transform,
        )
        self.n_fft = n_fft
        self.mel_scale = mel_scale
        self.spec_kwargs = {} if spec_kwargs is None else spec_kwargs
        self.spec_transform = spec_transform

    def process(self, idx: int) -> SpecMetadata:
        waveform, sample_rate, item = self._load_waveform(idx)

        if self.wav_transform is not None:
            waveform = self.wav_transform(waveform)

        spectrogram = compute_spectrogram(
            self.mel_scale, waveform, sample_rate, self.n_fft, self.spec_kwargs
        )

        if self.spec_transform is not None:
            spectrogram = self.spec_transform(spectrogram)

        return SpecMetadata(sample_id=item["time"], spectrogram=spectrogram)


class SpecProcessor(BaseProcessor):
    """Processes a single spectrogram, returns SpecMetadata."""

    def __init__(
        self,
        location: str,
        dataframe: pd.DataFrame,
        spec_transform: Optional[nn.Module | Compose] = None,
    ) -> None:
        super().__init__(location=location, dataframe=dataframe)
        self.spec_transform = spec_transform

    def process(self, idx: int) -> SpecMetadata:
        item = self.dataframe.iloc[idx]
        spectrogram = torch.load(
            os.path.join(self.location, item["filename"]), weights_only=True
        )

        assert isinstance(spectrogram, torch.Tensor)
        if spectrogram.ndim == 2:
            spectrogram = spectrogram.unsqueeze(0)

        if self.spec_transform is not None:
            spectrogram = self.spec_transform(spectrogram)

        return SpecMetadata(sample_id=item["time"], spectrogram=spectrogram)


class DualSpecProcessor(WavProcessor):
    """
    Processes a single waveform and spectrogram, returns a tuple of two SpecMetadata:
    - One without any transforms applied.
    - One with wav_transform and spec_transform applied.
    """

    def __init__(
        self,
        location: str,
        dataframe: pd.DataFrame,
        sample_rate: Optional[int] = None,
        add_channels: bool = True,
        n_fft: int = 1024,
        mel_scale: bool = True,
        spec_kwargs: Optional[dict[str, Any]] = None,
        wav_transform: Optional[nn.Module | Compose] = None,
        spec_transform: Optional[nn.Module | Compose] = None,
    ) -> None:
        super().__init__(
            location=location,
            dataframe=dataframe,
            sample_rate=sample_rate,
            add_channels=add_channels,
            wav_transform=wav_transform,
        )
        self.n_fft = n_fft
        self.mel_scale = mel_scale
        self.spec_kwargs = {} if spec_kwargs is None else spec_kwargs
        self.spec_transform = spec_transform

    def process(self, idx: int) -> tuple[SpecMetadata, SpecMetadata]:
        waveform, sample_rate, item = self._load_waveform(idx)
        spectrogram = compute_spectrogram(
            self.mel_scale, waveform, sample_rate, self.n_fft, self.spec_kwargs
        )

        if self.wav_transform is not None:
            waveform = self.wav_transform(waveform)

        spectrogram_proc = compute_spectrogram(
            self.mel_scale, waveform, sample_rate, self.n_fft, self.spec_kwargs
        )

        if self.spec_transform is not None:
            spectrogram_proc = self.spec_transform(spectrogram_proc)

        return (
            SpecMetadata(sample_id=item["time"], spectrogram=spectrogram),
            SpecMetadata(sample_id=item["time"], spectrogram=spectrogram_proc),
        )
