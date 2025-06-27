import math
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm


def copy_data(src_dir: str, dst_dir: str) -> None:
    """
    Copy files from src_dir to dst_dir, creating the directory structure in dst_dir.
    """
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    for file in tqdm(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, file)
        all_data = torch.load(src_path, weights_only=False)
        assert isinstance(all_data, dict)

        total += len(all_data)
        saved = 0
        pbar = tqdm(all_data.items(), total=len(all_data), leave=False)
        for filename, data in pbar:
            filename = os.path.basename(filename).replace(".wav", ".pt")
            dst_path = os.path.join(dst_dir, filename)
            if not os.path.exists(dst_path):
                spec = data["spec"]
                try:
                    assert isinstance(spec, torch.Tensor) and (spec.shape == (128, 65))
                    spec = (spec / math.log(10)).unsqueeze(0)
                    torch.save(spec, dst_path)
                    saved += 1
                    pbar.set_description(f"Saved {saved}/{len(all_data)}")
                    pbar.refresh()
                except AssertionError:
                    skipped += 1
                    continue
            else:
                saved += 1
                pbar.set_description(f"Saved {saved}/{len(all_data)}")
                pbar.refresh()

    print(f"Total files: {total}, Saved: {total - skipped}, Skipped: {skipped}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("src_dir", type=str, help="Source directory.")
    parser.add_argument("dst_dir", type=str, help="Destination directory.")
    args = parser.parse_args()

    copy_data(args.src_dir, args.dst_dir)
