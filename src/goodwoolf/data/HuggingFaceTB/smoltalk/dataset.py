"""Prepare SmolTalk for training in nanochat conversation format.

SmolTalk (``HuggingFaceTB/smol-smoltalk``) is a general conversational dataset,
used by SFT as the "general chat" part of the mixture. The records are
tokenizer-independent; the model tokenizer is applied at training time.

Each output line is one conversation::

    {"messages": [{"role": "system", "content": ...},
                  {"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}],
     "source": "smoltalk"}

``messages`` mirrors ``nanochat/tasks/smoltalk.py``: an optional system message
followed by alternating user/assistant turns, each with string content.

The fetched splits are preserved: each is written to its own ``<split>.parquet``
(``train.parquet``, ``test.parquet``). ``load_data(split)`` loads a single split and
``load_data()`` loads every split combined.

Usage:
    python -m goodwoolf.data.HuggingFaceTB.smoltalk.dataset [--fetch] [--prepare] [--split SPLIT] [--sample START COUNT]
"""

import sys
import json
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.common import record_text
from goodwoolf.utils.data import JsonDataset, prepare_json_splits, split_files
from goodwoolf.utils.system import dataset_path

DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# Raw parquet shards (the HF export) live under ``raw/<config>/<split>``.
SOURCE_DIRNAME = "raw"
PARQUET_REVISION = "refs/convert/parquet"

# The ``default`` config, as ``(config, split)`` pairs.
SOURCES = (
    ("default", "train"),
    ("default", "test"),
)

# Output split order; each is written to its own ``<split>.parquet``.
SPLITS = ("train", "test")


def fetch():
    """Download the SmolTalk parquet shards via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        allow_patterns=[f"{config}/{split}/*.parquet" for config, split in SOURCES],
        local_dir=dest,
    )
    for config, split in SOURCES:
        paths = sorted((dest / config / split).glob("*.parquet"))
        print(f"  {config}/{split}: {len(paths)} shard(s)", file=sys.stderr, flush=True)


def _conversation(row):
    """Build one nanochat conversation record from a SmolTalk row."""
    return {
        "messages": row["messages"],
        "source": "smoltalk",
    }


def prepare():
    """Write one ``<split>.parquet`` per fetched SmolTalk split."""
    prepare_json_splits(dataset_path(DATA_NAME), SOURCES, SPLITS, SOURCE_DIRNAME)


def load_data(split=None):
    """Return a JsonDataset for the prepared split(s).

    If ``split`` is given, load only ``<split>.parquet``; otherwise load every
    split in ``SPLITS`` order and combine them.
    """
    return JsonDataset(split_files(dataset_path(DATA_NAME), SPLITS, split), _conversation)


def metadata(split=None):
    """Print statistics for one split, or for every split when ``split`` is None."""
    for name in SPLITS if split is None else (split,):
        dataset = load_data(name)
        count = 0
        total_messages = 0
        total_chars = 0
        for record in dataset:
            count += 1
            total_messages += len(record["messages"])
            total_chars += len(record_text(record))

        print(f"File:            {name}.parquet")
        print(f"Conversations:   {count:,}")
        print(f"Messages:        {total_messages:,}")
        print(f"Total chars:     {total_chars:,}")
        if count:
            print(f"Avg chars/conv:  {total_chars / count:,.1f}")
        print(f"Format:          nanochat conversation messages (SFT chat)")


def sample(start=0, count=1, split=None):
    """Print COUNT samples starting at START, for one split or for every split."""
    for name in SPLITS if split is None else (split,):
        records = islice(load_data(name), start, start + count)
        print(f"Sample ({name}):")
        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare SmolTalk in nanochat conversation format")
    parser.add_argument("--fetch", action="store_true", help="Download the source dataset.")
    parser.add_argument("--prepare", action="store_true", help="Write the prepared dataset.")
    parser.add_argument("--metadata", action="store_true", help="Print dataset statistics and splits.")
    parser.add_argument("--split", default=None, help=f"Split to inspect: {', '.join(SPLITS)} (default: all).")
    parser.add_argument(
        "--sample",
        nargs=2,
        type=int,
        metavar=("START", "COUNT"),
        default=None,
        help="Print COUNT samples starting at START.",
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
