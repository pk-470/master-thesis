import os
from argparse import ArgumentParser
from pathlib import Path

from spec_mamba import clean_dataframe, get_GardenFiles23_df


def copy_metadata(data_dir: str, src_dir: str, csv_path: str) -> None:
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist.")

    filenames = os.listdir(data_dir)
    if not filenames:
        raise FileNotFoundError(f"No files found in {data_dir}.")
    print(f"Found {len(filenames)} spectrograms in {data_dir}.")

    print("Loading metadata...")
    all_metadata = get_GardenFiles23_df(src_dir)

    print("Processing metadata...")
    all_metadata = clean_dataframe(all_metadata)
    available_times = set(
        x.replace("er_file", "").replace(".pt", "") for x in filenames
    )
    common = set(all_metadata["time"]).intersection(available_times)
    all_metadata = all_metadata[all_metadata["time"].isin(common)].reset_index(
        drop=True
    )
    all_metadata["filename"] = all_metadata["time"].apply(
        lambda x: f"GardenFiles23/er_file{x}.pt"
    )
    print("Dataframe length:", len(all_metadata))

    print("Saving metadata...")
    all_metadata.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory.")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file path.")
    args = parser.parse_args()

    copy_metadata(data_dir=args.data_dir, src_dir=args.src_dir, csv_path=args.csv_path)
