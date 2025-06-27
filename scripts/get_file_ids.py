"""Get DataverseNL file ids from the metadata.json."""

import argparse
import json
import os
import re


def parse_input_indices(input_indices_str: str, files: list[dict]) -> list[str]:
    """Parse indices from input."""
    if "all" in input_indices_str:
        return [str(i) for i in range(len(files))]
    indices = []
    if matches := re.findall(r"\d+-\d+", input_indices_str):
        for match in matches:
            start, end = match.split("-")
            indices.extend([str(i) for i in range(int(start), int(end) + 1)])
            input_indices_str = input_indices_str.replace(match, "")
    if re.search(r"\d+(\s\d+)*", input_indices_str):
        indices.extend(input_indices_str.split())

    return list(sorted(set(indices), key=int))


def get_dir_indices() -> list[str]:
    """Get indices of already downloaded subdirectories."""
    dir_indices = []
    for dir_name in os.listdir():
        if (match := re.search(r"GardenFiles23_(\d+)", dir_name)) is not None:
            dir_indices.append(match.group(1))

    return dir_indices


def get_file_ids(files: list[dict], zip_indices: list[str]) -> str:
    """Get file ids corresponding to zip_indices from list of files."""
    file_ids = []
    for file in files:
        if (
            (match := re.search(r"GardenFiles23_(\d+).zip", file["label"])) is not None
        ) and match.group(1) in zip_indices:
            file_ids.append(str(file["dataFile"]["id"]))

    return " ".join(file_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "zip_indices",
        nargs="+",
        type=str,
        help='Accepted args: e.g. "0-2" (inclusive range), "0 1" (exact), "all" (all data)',
    )
    args = parser.parse_args()
    indices_str: str = " ".join(args.zip_indices)

    if not os.path.exists("metadata.json"):
        raise OSError("metadata.json does not exist")

    with open("metadata.json", "r", encoding="utf8") as f:
        metadata = json.load(f)
    all_files = metadata["datasetVersion"]["files"]

    input_indices = parse_input_indices(indices_str, all_files)
    new_indices = list(filter(lambda x: x not in get_dir_indices(), input_indices))
    new_file_ids = get_file_ids(all_files, new_indices)

    print(new_file_ids)
