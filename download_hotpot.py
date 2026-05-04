#!/usr/bin/env python3
"""Download HotpotQA from Hugging Face to local disk."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, DatasetDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download hotpotqa/hotpot_qa dataset")
    parser.add_argument(
        "--subset",
        choices=["distractor", "fullwiki", "both"],
        default="both",
        help="Subset to download. Default: distractor",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save dataset files",
    )
    return parser.parse_args()


def save_subset(name: str, output_dir: Path) -> None:
    print(f"Downloading subset: {name}")
    ds: DatasetDict = load_dataset("hotpotqa/hotpot_qa", name=name)
    subset_dir = output_dir / name
    subset_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(subset_dir))
    print(f"Saved {name} to {subset_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["distractor", "fullwiki"] if args.subset == "both" else [args.subset]
    for subset in subsets:
        save_subset(subset, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
