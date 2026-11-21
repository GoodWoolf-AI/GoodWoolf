"""Prepare KL3M USC as parquet text.

The source dataset stores KL3M token ids instead of raw text. ``prepare``
decodes those tokens with the KL3M tokenizer and writes ``train.parquet`` and
``test.parquet`` with one document per row in the ``text`` column. The single
source shard is split by document count, with the final 20% used as the test
split. Tokenization happens again at training time with the model tokenizer.

Usage:
    goodwoolf dataset alea-institute/kl3m-usc [--fetch] [--prepare]
                               [--sample START COUNT]

Writes ``train.parquet`` and ``test.parquet`` under
``<GOODWOOLF_DATA>/alea-institute/kl3m-usc``.
"""

import sys
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.data import (
    TextDataset,
    count_parquet_rows,
    split_files,
    split_source_texts,
    text_metadata,
    write_text_splits,
)
from goodwoolf.utils.system import dataset_path, ensure_dir

DATASET_NAME = "alea-institute/kl3m-data-usc"
TOKENIZER_NAME = "alea-institute/kl3m-004-128k-cased"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])
SOURCE_DIRNAME = "raw"
SOURCE_DIR = "data"
SOURCE_PATTERN = "train-*.parquet"
SPLITS = ("train", "test")
TRAIN_RATIO = 0.8


def _source_paths(source_dir):
    return sorted((source_dir / SOURCE_DIR).glob(SOURCE_PATTERN))


def fetch():
    """Download the KL3M USC source parquet via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        allow_patterns=[f"{SOURCE_DIR}/{SOURCE_PATTERN}"],
        local_dir=dest,
    )
    paths = _source_paths(dest)
    print(f"  {len(paths)} parquet shard(s) -> {dest / SOURCE_DIR}", file=sys.stderr, flush=True)


def _texts(paths):
    """Decode KL3M token ids into documents."""
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    for path in paths:
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.num_row_groups):
            tokens = parquet.read_row_group(row_group, columns=["tokens"]).column("tokens").to_pylist()
            for token_ids in tokens:
                if not token_ids:
                    continue
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                if text:
                    yield text


def prepare():
    """Decode KL3M tokens into train.parquet and test.parquet."""
    data_dir = ensure_dir(dataset_path(DATA_NAME))
    source_dir = data_dir / SOURCE_DIRNAME
    raw_paths = _source_paths(source_dir)
    if not raw_paths:
        raise FileNotFoundError(f"No {SOURCE_PATTERN} in {source_dir / SOURCE_DIR}; run with --fetch first")

    train_texts, test_texts = split_source_texts(
        raw_paths, _texts, count_parquet_rows, TRAIN_RATIO
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
    parser = argparse.ArgumentParser(description="Prepare KL3M USC as parquet text")
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
