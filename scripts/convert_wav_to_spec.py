"""Script to run the waveform to spectrogram converter."""

if __name__ == "__main__":
    import argparse

    import torchaudio.transforms as T

    from spec_mamba import (
        SPECTROGRAMS_1C_LOCATION,
        SPECTROGRAMS_2C_LOCATION,
        WAVEFORMS_DIR,
        WAVEFORMS_LOCATION,
        WavToSpecConverter,
        create_paths,
        get_GardenFiles23_df,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "channels",
        type=int,
        help="Number of channels (1 or 2).",
    )
    args = parser.parse_args()
    channels = args.channels

    if channels not in (1, 2):
        raise ValueError("Number of channels can be 1 or 2.")

    create_paths()

    print("Loading dataframe...")
    dataframe = get_GardenFiles23_df(WAVEFORMS_DIR)

    print(f"Converting waveforms to spectrograms ({channels=})...")
    converter = WavToSpecConverter(
        wav_location=WAVEFORMS_LOCATION,
        spec_location=(
            SPECTROGRAMS_1C_LOCATION if (channels == 1) else SPECTROGRAMS_2C_LOCATION
        ),
        dataframe=dataframe,
        sample_rate=48_000,
        add_channels=(channels == 1),
        n_fft=4096,
        mel_scale=True,
        spec_kwargs={
            "n_mels": 128,
        },
        spec_transform=T.AmplitudeToDB(),
    )
    converter.convert_and_save(overwrite=True)
