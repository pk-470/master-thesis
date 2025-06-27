"""GardenFiles23 dataset."""

from typing import Callable, Optional

import numpy as np
from numpy.typing import DTypeLike
from torch import Tensor
from torch.utils.data import Dataset

from spec_mamba.data.metadata import *
from spec_mamba.data.processors import *


class GF23Dataset(Dataset):
    """GardenFiles23 dataset."""

    def __init__(
        self,
        processor: WavToSpecProcessor | SpecProcessor,
        target_fn: Optional[Callable[[Tensor], Tensor]] = None,
        labels: Optional[str | list[str]] = None,
        labels_dtype: Optional[DTypeLike] = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        if (target_fn is not None) and (labels is not None):
            raise ValueError(
                "'target_fn' and 'labels' cannot be defined simultaneously."
            )
        if (labels is not None) and (labels_dtype is None):
            raise ValueError("Specify the data type for the labels.")
        self.target_fn = target_fn
        self.labels = labels
        self.labels_dtype = labels_dtype

    def __len__(self) -> int:
        return len(self.processor.dataframe)

    def _add_target(self, sample_dict: SpecMetadata, idx: int) -> SpecSample:
        sample_dict = SpecSample(**sample_dict)

        if self.target_fn is not None:
            sample_dict["target"] = self.target_fn(sample_dict["spectrogram"])

        if self.labels is not None:
            assert self.labels_dtype is not None
            if isinstance(self.labels, str):
                value = self.processor.dataframe.at[idx, self.labels]
                sample_dict["target"] = np.asarray(value, dtype=self.labels_dtype)
            else:
                sample_dict["target"] = (
                    self.processor.dataframe[self.labels]
                    .iloc[idx]
                    .to_numpy(dtype=self.labels_dtype)
                )

        return sample_dict

    def __getitem__(self, idx: int) -> SpecSample:
        sample_dict = self.processor.process(idx)
        sample_dict = self._add_target(sample_dict, idx=idx)

        return sample_dict


class DualSpecGF23Dataset(Dataset):
    """
    GardenFiles23 dataset for DualSpecProcessor, returns a tuple of two SpecSample objects:
    - One without any transforms applied.
    - One with wav_transform and spec_transform applied.
    """

    def __init__(self, processor: DualSpecProcessor) -> None:
        super().__init__()
        self.processor = processor

    def __len__(self) -> int:
        return len(self.processor.dataframe)

    def __getitem__(self, idx: int) -> tuple[SpecSample, SpecSample]:
        sample_dict_orig, sample_dict_proc = self.processor.process(idx)
        sample_dict_orig = SpecSample(**sample_dict_orig)
        sample_dict_proc = SpecSample(**sample_dict_proc)

        return sample_dict_orig, sample_dict_proc
