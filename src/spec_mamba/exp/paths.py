"""Default paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base
CHECKPOINTS_LOCATION = os.getenv("CHECKPOINTS_LOCATION", "checkpoints")
DATA_LOCATION = os.getenv("DATA_LOCATION", "dataset")

# Waveforms
WAVEFORMS_LOCATION = os.path.join(DATA_LOCATION, "waveforms")
WAVEFORMS_DIR = os.path.join(WAVEFORMS_LOCATION, "GardenFiles23")
WAV_SPLITS_DIR = os.path.join(WAVEFORMS_LOCATION, "splits")
WAV_FOUNDATION_SPLITS_DIR = os.path.join(WAV_SPLITS_DIR, "foundation")
WAV_TARGETS_SPLITS_DIR = os.path.join(WAV_SPLITS_DIR, "targets")

# Spectrograms 1 channel
SPECTROGRAMS_1C_LOCATION = os.path.join(DATA_LOCATION, "spectrograms", "one_channel")
SPECTROGRAMS_1C_DIR = os.path.join(SPECTROGRAMS_1C_LOCATION, "GardenFiles23")
SPEC_1C_SPLITS_DIR = os.path.join(SPECTROGRAMS_1C_LOCATION, "splits")
SPEC_1C_FOUNDATION_SPLITS_DIR = os.path.join(SPEC_1C_SPLITS_DIR, "foundation")
SPEC_1C_TARGETS_SPLITS_DIR = os.path.join(SPEC_1C_SPLITS_DIR, "targets")

# Spectrograms 2 channels
SPECTROGRAMS_2C_LOCATION = os.path.join(DATA_LOCATION, "spectrograms", "two_channels")
SPECTROGRAMS_2C_DIR = os.path.join(SPECTROGRAMS_2C_LOCATION, "GardenFiles23")
SPEC_2C_FOUNDATION_SPLITS_DIR = os.path.join(SPECTROGRAMS_2C_LOCATION, "splits")
SPEC_2C_FOUNDATION_SPLITS_DIR = os.path.join(
    SPEC_2C_FOUNDATION_SPLITS_DIR, "foundation"
)
SPEC_2C_TARGETS_SPLITS_DIR = os.path.join(SPEC_2C_FOUNDATION_SPLITS_DIR, "targets")

# SpecData
SPEC_DATA_LOCATION = os.path.join(DATA_LOCATION, "specData")
SPEC_DATA_DIR = os.path.join(SPEC_DATA_LOCATION, "GardenFiles23")
SPEC_DATA_SPLITS_DIR = os.path.join(SPEC_DATA_LOCATION, "splits")
SPEC_DATA_FOUNDATION_SPLITS_DIR = os.path.join(SPEC_DATA_SPLITS_DIR, "foundation")
SPEC_DATA_TARGETS_SPLITS_DIR = os.path.join(SPEC_DATA_SPLITS_DIR, "targets")

# Splits
SPLITS_DIR = os.path.join(DATA_LOCATION, "splits")
FOUNDATION_SPLITS_DIR = os.path.join(SPLITS_DIR, "foundation")
CONTRASTIVE_SPLITS_DIR = os.path.join(SPLITS_DIR, "contrastive")
TARGETS_SPLITS_DIR = os.path.join(SPLITS_DIR, "targets")

PATHS = (
    # Base
    CHECKPOINTS_LOCATION,
    DATA_LOCATION,
    # Waveforms
    WAVEFORMS_LOCATION,
    WAVEFORMS_DIR,
    WAV_SPLITS_DIR,
    WAV_FOUNDATION_SPLITS_DIR,
    WAV_TARGETS_SPLITS_DIR,
    # Spectrograms 1 channel
    SPECTROGRAMS_1C_LOCATION,
    SPECTROGRAMS_1C_DIR,
    SPEC_1C_SPLITS_DIR,
    SPEC_1C_FOUNDATION_SPLITS_DIR,
    SPEC_1C_TARGETS_SPLITS_DIR,
    # Spectrograms 2 channels
    SPECTROGRAMS_2C_LOCATION,
    SPECTROGRAMS_2C_DIR,
    SPEC_2C_FOUNDATION_SPLITS_DIR,
    SPEC_2C_FOUNDATION_SPLITS_DIR,
    SPEC_2C_TARGETS_SPLITS_DIR,
    # SpecData
    SPEC_DATA_LOCATION,
    SPEC_DATA_DIR,
    SPEC_DATA_SPLITS_DIR,
    SPEC_DATA_FOUNDATION_SPLITS_DIR,
    SPEC_DATA_TARGETS_SPLITS_DIR,
    # Splits
    SPLITS_DIR,
    FOUNDATION_SPLITS_DIR,
    CONTRASTIVE_SPLITS_DIR,
    TARGETS_SPLITS_DIR,
)


def create_paths() -> None:
    for path in PATHS:
        Path(path).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_paths()
