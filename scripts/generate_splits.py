"""Generate train/validation/test splits."""

import glob
import os

import pandas as pd

from spec_mamba import (
    SPEC_1C_FOUNDATION_SPLITS_DIR,
    SPEC_2C_FOUNDATION_SPLITS_DIR,
    SPECTROGRAMS_1C_DIR,
    SPECTROGRAMS_2C_DIR,
    WAV_FOUNDATION_SPLITS_DIR,
    WAVEFORMS_DIR,
    clean_dataframe,
    create_paths,
    get_GardenFiles23_df,
    train_val_test_split,
)


def _train_val_test_split(
    splits_dir: str,
    dataframe: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
    stratify=None,
) -> pd.DataFrame:

    train_df, val_df, test_df = train_val_test_split(
        dataframe,
        train_size=train_size,
        val_size=val_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    results = {}

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = os.path.join(splits_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        results[name] = {"Path": path, "Samples": len(df)}

    return pd.DataFrame.from_dict(results, orient="index")


if __name__ == "__main__":
    create_paths()
    pd.set_option("display.max_colwidth", None)

    print("Loading waveform dataframe...")
    wav_metadata = clean_dataframe(get_GardenFiles23_df(WAVEFORMS_DIR))
    wav_splits_results = _train_val_test_split(
        WAV_FOUNDATION_SPLITS_DIR, wav_metadata, train_size=0.9, val_size=0.05
    )
    print("----- Waveform foundation splits -----")
    print(wav_splits_results)
    print()

    for name, spec_dir, splits_dir in (
        ("one", SPECTROGRAMS_1C_DIR, SPEC_1C_FOUNDATION_SPLITS_DIR),
        ("two", SPECTROGRAMS_2C_DIR, SPEC_2C_FOUNDATION_SPLITS_DIR),
    ):
        if not glob.glob(os.path.join(spec_dir, "*.csv")):
            print(
                f"Skipping {name} channel spectrogram split: no metadata found in {spec_dir}."
            )
            print()
        else:
            print(f"Loading {name} channel spectrogram dataframe...")
            spec_metadata = clean_dataframe(
                pd.read_csv(glob.glob(os.path.join(spec_dir, "*.csv"))[0])
            )
            spec_splits_results = _train_val_test_split(
                splits_dir, spec_metadata, train_size=0.9, val_size=0.05
            )
            print(
                f"----- {name.capitalize()} channel spectrogram foundation splits -----"
            )
            print(spec_splits_results)
            print()
