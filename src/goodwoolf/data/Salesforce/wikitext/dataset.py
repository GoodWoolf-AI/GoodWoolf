"""Prepare WikiText-2 as parquet text.

WikiText-2 is a small language-modeling corpus with train, validation, and
test splits. Each split is stored as parquet with one document per row in the
``text`` column. Tokenization happens at training time.

Usage:
    python -m goodwoolf.data.Salesforce.wikitext.dataset [--fetch] [--prepare]
                                              [--sample START COUNT]

Writes ``train.parquet``, ``validation.parquet``, and ``test.parquet`` under
``<GOODWOOLF_DATA>/Salesforce/wikitext``.
"""

import sys
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.data import (
    TextDataset,
    read_parquet_texts,
    split_files,
    text_metadata,
    write_text_splits,
)
from goodwoolf.utils.system import dataset_path, ensure_dir

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])
SOURCE_DIRNAME = "raw"
SPLITS = ("train", "validation", "test")


def _source_paths(source_dir, split):
    return sorted((source_dir / DATASET_CONFIG).glob(f"{split}-*.parquet"))


def fetch():
    """Download the WikiText-2 raw parquet files via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        allow_patterns=[f"{DATASET_CONFIG}/*.parquet"],
        local_dir=dest,
    )
    for split in SPLITS:
        paths = _source_paths(dest, split)
        print(f"  {split}: {len(paths)} shard(s)", file=sys.stderr, flush=True)


def prepare():
    """Write train.parquet, validation.parquet, and test.parquet."""
    data_dir = ensure_dir(dataset_path(DATA_NAME))
    source_dir = data_dir / SOURCE_DIRNAME
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} not found; run with --fetch first")

    splits = {}
    for split in SPLITS:
        paths = _source_paths(source_dir, split)
        if not paths:
            raise FileNotFoundError(f"No WikiText-2 source files for {split}; run with --fetch first")
        splits[split] = read_parquet_texts(paths)

    write_text_splits(data_dir, splits)


def load_data(split=None):
    return TextDataset(files=split_files(dataset_path(DATA_NAME), SPLITS, split))


def metadata(split=None):
    text_metadata(split_files(dataset_path(DATA_NAME), SPLITS, split))
    print("Format: parquet text column (tokenized at training time)")


def sample(start=0, count=1, split=None):
    for name in SPLITS if split is None else (split,):
        files = split_files(dataset_path(DATA_NAME), SPLITS, name)

        print(f"Sample ({name}):")
        for line in islice(TextDataset(files=files), start, start + count):
            print(line)
            print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 as parquet text")
    parser.add_argument("--fetch", action="store_true", help="Download the source dataset.")
    parser.add_argument("--prepare", action="store_true", help="Write text parquet split files.")
    parser.add_argument("--metadata", action="store_true", help="Print dataset statistics and splits.")
    parser.add_argument(
        "--sample",
        nargs=2,
        type=int,
        metavar=("START", "COUNT"),
        default=None,
        help="Print COUNT samples starting at START.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=f"Split to inspect: {', '.join(SPLITS)} (default: all).",
    )
    args = parser.parse_args(argv)

    sample_requested = args.sample is not None
    no_action = not (args.fetch or args.prepare or args.metadata or sample_requested)
    if no_action:
        args.fetch = True
        args.prepare = True

    if args.fetch:
        fetch()
    if args.prepare:
        prepare()
    if args.prepare or args.metadata:
        metadata(args.split)
    if sample_requested:
        sample(*args.sample, split=args.split)
    elif no_action:
        sample(0, 1, split=args.split)


if __name__ == "__main__":
    main()
