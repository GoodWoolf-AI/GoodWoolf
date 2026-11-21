"""Prepare Nemotron-ClimbMix as parquet text.

The output is tokenizer-independent: ``prepare`` detokenizes the source JSONL
shards and writes ``train.parquet`` and ``test.parquet`` with one document per
row in the ``text`` column; tokenization happens at training time.

``nvidia/Nemotron-ClimbMix`` ships GPT-2 token ids (the ``tokens`` field of
each ``part_*.tokenized.jsonl`` record), so ``prepare`` detokenizes them back
to text with tiktoken's ``gpt2`` encoding. The source files are split by file:
the last file becomes the test split and all earlier files form the train
split.

Usage:
    python -m goodwoolf.data.nvidia.climbmix.dataset [--fetch] [--prepare]
                                              [--sample START COUNT]

Writes ``train.parquet`` and ``test.parquet`` under
``<GOODWOOLF_DATA>/nvidia/climbmix``.
"""

import sys
import json
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.data import (
    TextDataset,
    count_lines,
    split_files,
    split_source_texts,
    text_metadata,
    write_text_splits,
)
from goodwoolf.utils.system import dataset_path, ensure_dir

NUM_PROC = 8
DATASET_NAME = "nvidia/Nemotron-ClimbMix"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# The full 400B-token dataset is shipped as tokenized JSONL shards on main.
SOURCE_PATTERN = "part_*.tokenized.jsonl"
SOURCE_DIRNAME = "raw"
SPLITS = ("train", "test")
TRAIN_RATIO = 0.8
ENCODING = "gpt2"


def fetch():
    """Download the tokenized ClimbMix JSONL shards via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        allow_patterns=SOURCE_PATTERN,
        local_dir=dest,
        max_workers=NUM_PROC,
    )
    paths = sorted(dest.glob(SOURCE_PATTERN))
    print(f"  {len(paths)} jsonl shard(s) -> {dest}", file=sys.stderr, flush=True)


def _texts(raw_paths, encoding):
    """Yield text by decoding the GPT-2 token ids of each tokenized JSONL record."""
    for path in raw_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = json.loads(line).get("tokens", [])
                if tokens:
                    yield encoding.decode(tokens)


def prepare():
    """Detokenize the ClimbMix GPT-2 tokens into train.parquet and test.parquet."""
    import tiktoken

    data_dir = ensure_dir(dataset_path(DATA_NAME))
    source_dir = data_dir / SOURCE_DIRNAME
    raw_paths = sorted(source_dir.glob(SOURCE_PATTERN))
    if not raw_paths:
        raise FileNotFoundError(f"No {SOURCE_PATTERN} in {source_dir}; run with --fetch first")

    encoding = tiktoken.get_encoding(ENCODING)

    print(f"Detokenizing {len(raw_paths)} source file(s) ...", file=sys.stderr, flush=True)
    train_texts, test_texts = split_source_texts(
        raw_paths,
        lambda paths: _texts(paths, encoding),
        count_lines,
        TRAIN_RATIO,
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
    parser = argparse.ArgumentParser(description="Prepare Nemotron-ClimbMix as multi-line text")
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
