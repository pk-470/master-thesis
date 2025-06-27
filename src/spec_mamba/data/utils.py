"""Data utils."""

import glob
import math
import os
from typing import Any, Literal, Optional, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split
from torch import Tensor


def get_GardenFiles23_df(dir: str) -> pd.DataFrame:
    """Load, clean and concatenate all Excel files (.xlsx) found in the given directory and its subfolders."""
    dfs = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(".xlsx"):
                file_path = os.path.join(root, file)
                df = pd.read_excel(file_path, index_col=0)
                df = df[df["filename"] != "file removed"]
                dfs.append(df)
    dataframe = pd.concat(dfs).reset_index(drop=True)

    return dataframe


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Applies dropna, drop_duplicates, reset_index."""
    return (
        dataframe.drop_duplicates(subset="time", keep=False)
        .reset_index(drop=True)
        .convert_dtypes()
    )


def get_split_df(splits_dir: str, mode: str) -> pd.DataFrame:
    """Get the train/validation/test dataframes."""
    csv_files = glob.glob(os.path.join(splits_dir, f"{mode}*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No csv file starting with {mode} was found in {splits_dir}"
        )
    dataframe = pd.read_csv(csv_files[0], index_col=0)

    return dataframe


def get_number_of_steps(
    splits_dir: str, batch_size: int = 64, num_devices: int = 1
) -> int:
    """
    Calculate the number of steps per epoch for the dataset based on
    the batch size and the number of GPUs.
    """
    train_df = get_split_df(splits_dir, mode="train")
    return int(math.ceil(len(train_df) / batch_size / num_devices))


def get_label_weights(splits_dir: str, label: str) -> Tensor:
    """Calculate the weight of each label as the inverse class frequency."""
    train_df = get_split_df(splits_dir, mode="train")
    counts = train_df[label].value_counts().sort_index().to_numpy()
    weights = len(train_df) / counts
    return torch.from_numpy(weights).type(torch.float)


def train_val_test_split(
    dataframe: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
    stratify=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split using train_test_split from sklearn."""

    dataframe_train, dataframe_val_test = train_test_split(
        dataframe,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    dataframe_val, dataframe_test = train_test_split(
        dataframe_val_test,
        test_size=val_size / (1 - train_size),
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    return dataframe_train, dataframe_val, dataframe_test


def plot_waveform(
    waveform: Tensor, sample_rate: int, sample_id: Optional[str] = None, **kwargs
) -> None:
    """Plot waveform."""
    waveform_: np.ndarray = waveform.numpy()
    num_channels, num_frames = waveform_.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)

    if isinstance(axes, Axes):
        axes = [axes]

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform_[c], linewidth=1)
        axes[c].grid(True)

        axes[c].set_xlabel("Time (s)")

        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        else:
            axes[c].set_ylabel("Amplitude")

    if sample_id is not None:
        figure.suptitle(sample_id)

    plt.tight_layout()


def compute_spectrogram(
    mel_scale: bool,
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    spec_kwargs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Compute a spectrogram or mel spectrogram."""
    if spec_kwargs is None:
        spec_kwargs = {}

    if mel_scale:
        spec_fn = T.MelSpectrogram(sample_rate, n_fft=n_fft, **spec_kwargs)
    else:
        spec_fn = T.Spectrogram(n_fft=n_fft, **spec_kwargs)

    return spec_fn(waveform)


def plot_spectrogram(
    spectrogram: Tensor, sample_id: Optional[str] = None, to_db: bool = True, **kwargs
) -> None:
    """Plot spectrogram."""
    spectrogram_ = spectrogram.numpy()
    num_channels = spectrogram_.shape[0]

    figure, axes = plt.subplots(num_channels, 1)

    if isinstance(axes, Axes):
        axes = [axes]

    for c in range(num_channels):
        spec_plot = librosa.power_to_db(spectrogram_[c]) if to_db else spectrogram_[c]
        plot = axes[c].imshow(
            spec_plot,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        cbar = plt.colorbar(plot)
        cbar.set_label("dB" if to_db else "Power")

        axes[c].set_xlabel("Window Index")

        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        else:
            axes[c].set_ylabel("Frequency Bin")

    if sample_id is not None:
        figure.suptitle(sample_id)

    plt.tight_layout()


def get_mask(
    shape: Sequence[int],
    mask_ratio: float,
    device: Any = "cpu",
    masking_mode: Literal["unstructured", "timestep"] = "unstructured",
    grid_width: Optional[int] = None,
) -> Tensor:
    """Generate a binary mask (0: keep, 1: mask) for the input tensor."""
    if (mask_ratio < 0) or (mask_ratio > 1):
        raise ValueError("mask_ratio should be in [0, 1].")
    if masking_mode not in ("unstructured", "timestep"):
        raise ValueError("masking_mode can be one of: 'unstructured', 'timestep'.")

    N, L = shape[:2]
    len_keep = L - int(L * mask_ratio)

    if masking_mode == "unstructured":
        noise = torch.rand(N, L, device=device)
    else:
        if grid_width is None:
            raise ValueError("grid_width must be provided for 'timestep' masking mode.")
        noise = torch.rand(N, L // grid_width, device=device)
        noise = torch.repeat_interleave(noise, repeats=grid_width, dim=1)

    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=device)

    # keep the first subset
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return mask
