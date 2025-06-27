"""
WavToSpecConverter: Converts all waveforms to spectrograms with the chosen parameters,
saves them as .pt files in the chosen location and saves their metadata.
"""

import os
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import Module
from tqdm.auto import tqdm

from spec_mamba.data.processors import WavToSpecProcessor


class WavToSpecConverter(WavToSpecProcessor):
    """
    Converts all waveforms to spectrograms with the chosen parameters,
    saves them as .pt files in the chosen location, along with their metadata
    in a .csv file in the same location.
    """

    def __init__(
        self,
        wav_location: str,
        spec_location: str,
        dataframe: DataFrame,
        sample_rate: Optional[int] = None,
        add_channels: bool = True,
        n_fft: int = 400,
        mel_scale: bool = True,
        spec_kwargs: Optional[dict[str, Any]] = None,
        wav_transform: Optional[Module] = None,
        spec_transform: Optional[Module] = None,
    ) -> None:
        super().__init__(
            location=wav_location,
            dataframe=dataframe,
            sample_rate=sample_rate,
            add_channels=add_channels,
            n_fft=n_fft,
            mel_scale=mel_scale,
            spec_kwargs=spec_kwargs,
            wav_transform=wav_transform,
            spec_transform=spec_transform,
        )
        self.spec_location = spec_location
        self.metadata_path = os.path.join(self.spec_location, "spec_metadata.csv")

        Path(os.path.join(spec_location, "GardenFiles23")).mkdir(
            parents=True, exist_ok=True
        )

    def convert_and_save(
        self, indices: Optional[Iterable[int]] = None, overwrite: bool = False
    ):
        overwrite = overwrite or not os.path.exists(self.metadata_path)

        if indices is None:
            if overwrite:
                indices = range(len(self.dataframe))
            else:
                current_metadata = pd.read_csv(self.metadata_path, index_col=0)
                indices = self.dataframe.loc[
                    ~self.dataframe["time"].isin(current_metadata["time"])
                ].index.tolist()

        new_filenames = []

        for idx in tqdm(indices):
            spec_metadata = self.process(idx)
            filename = f"GardenFiles23/er_file{spec_metadata['sample_id']}.pt"
            torch.save(
                spec_metadata["spectrogram"], os.path.join(self.spec_location, filename)
            )
            new_filenames.append(filename)

        if overwrite:
            metadata = self.dataframe.loc[list(indices)].copy()
            filenames = new_filenames
        else:
            metadata = self.dataframe.copy()
            filenames = []
            new_filenames_iter = iter(new_filenames)
            old_filenames_iter = iter(current_metadata["filename"])
            for idx in range(len(self.dataframe)):
                if idx in indices:
                    filenames.append(next(new_filenames_iter))
                else:
                    filenames.append(next(old_filenames_iter))

        metadata["filename"] = filenames
        metadata.to_csv(self.metadata_path, index=False)
