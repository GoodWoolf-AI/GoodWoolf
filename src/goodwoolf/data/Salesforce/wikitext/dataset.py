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


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Prepare WikiText-2 as parquet text")
    actions = parser.add_subparsers(dest="action")

    actions.add_parser("fetch", help="Download the source dataset.")
    actions.add_parser("prepare", help="Write text parquet split files.")

    metadata_parser = actions.add_parser("metadata", help="Print dataset statistics and splits.")
    metadata_parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help=f"Split to inspect: {', '.join(SPLITS)} (default: all).",
    )

    sample_parser = actions.add_parser("sample", help="Print COUNT samples starting at START.")
    sample_parser.add_argument("start", nargs="?", type=int, default=0)
    sample_parser.add_argument("count", nargs="?", type=int, default=1)
    sample_parser.add_argument(
        "--split",
        choices=SPLITS,
        default=None,
        help=f"Split to inspect: {', '.join(SPLITS)} (default: all).",
    )

    args = parser.parse_args(argv)
    if args.action is None:
        parser.print_help()
        return
    if args.action == "fetch":
        fetch()
    elif args.action == "prepare":
        prepare()
    elif args.action == "metadata":
        metadata(args.split)
    else:
        sample(args.start, args.count, split=args.split)


if __name__ == "__main__":
    main()
