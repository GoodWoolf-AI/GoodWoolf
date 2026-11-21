"""Prepare OpenWebText as parquet text.

The output is tokenizer-independent: ``prepare`` writes ``train.parquet`` and
``test.parquet`` with one document per row in the ``text`` column, and
tokenization happens at training time. The raw parquet shards are split by
file: the last shard becomes the test split and all earlier shards form the
train split.

Usage:
    python -m goodwoolf.data.Skylion007.openwebtext.dataset [--fetch] [--prepare]
                                                 [--sample START COUNT]

Writes ``train.parquet`` and ``test.parquet`` under
``<GOODWOOLF_DATA>/Skylion007/openwebtext``.
"""

import sys
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.data import (
    TextDataset,
    count_parquet_rows,
    read_parquet_texts,
    split_files,
    split_source_texts,
    text_metadata,
    write_text_splits,
)
from goodwoolf.utils.system import dataset_path, ensure_dir

NUM_PROC = 8
DATASET_NAME = "Skylion007/openwebtext"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# Raw parquet shards (the HF export) live under ``raw/<config>/<split>``.
SOURCE_DIRNAME = "raw"
SOURCE_CONFIG = "plain_text"
SOURCE_SPLIT = "train"
SPLITS = ("train", "test")
TRAIN_RATIO = 0.8

# Every Hub dataset has an auto-generated parquet export on this revision.
PARQUET_REVISION = "refs/convert/parquet"


def fetch():
    """Download the OpenWebText parquet shards via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        allow_patterns=[f"{SOURCE_CONFIG}/{SOURCE_SPLIT}/*.parquet"],
        local_dir=dest,
        max_workers=NUM_PROC,
    )
    paths = sorted((dest / SOURCE_CONFIG / SOURCE_SPLIT).glob("*.parquet"))
    print(f"  {SOURCE_CONFIG}/{SOURCE_SPLIT}: {len(paths)} shard(s)", file=sys.stderr, flush=True)


def prepare():
    """Write ``train.parquet`` and ``test.parquet`` from the downloaded shards."""
    data_dir = ensure_dir(dataset_path(DATA_NAME))
    source_dir = data_dir / SOURCE_DIRNAME
    if not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} not found; run with --fetch first")

    raw_paths = sorted((source_dir / SOURCE_CONFIG / SOURCE_SPLIT).glob("*.parquet"))
    if not raw_paths:
        raise FileNotFoundError(f"No source shards found in {source_dir}; run with --fetch first")

    train_texts, test_texts = split_source_texts(
        raw_paths, read_parquet_texts, count_parquet_rows, TRAIN_RATIO
    )
    write_text_splits(data_dir, dict(zip(SPLITS, (train_texts, test_texts))))


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
    parser = argparse.ArgumentParser(description="Prepare OpenWebText as multi-line text")
    parser.add_argument("--fetch", action="store_true", help="Download the source dataset.")
    parser.add_argument("--prepare", action="store_true", help="Write text parquet to train.parquet and test.parquet.")
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
