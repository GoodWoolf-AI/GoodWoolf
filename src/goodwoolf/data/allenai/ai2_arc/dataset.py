"""Prepare ARC (AI2 Reasoning Challenge) for training in GoodWoolf conversation format.

ARC is a multiple-choice science benchmark. It is prepared once and used by SFT
(and evaluation). The records are tokenizer-independent; the model tokenizer is
applied at training time.

Each output line is one example::

    {"messages": [{"role": "user", "content": "Multiple Choice question: ..."},
                  {"role": "assistant", "content": "B"}],
     "answer": "B",
     "letters": ["A", "B", "C", "D"],
     "source": "ai2_arc"}

``messages`` uses GoodWoolf's conversation format: the user message renders the
question with its choices via ``render_mc`` and the assistant message is the
answer letter.

The fetched splits are preserved: each is written to its own ``<split>.parquet``
(``train.parquet``, ``validation.parquet``, ``test.parquet``). ``load_data(split)``
loads a single split and ``load_data()`` loads every split combined.

Usage:
    python -m goodwoolf.data.allenai.ai2_arc.dataset [--fetch] [--prepare] [--split SPLIT] [--sample START COUNT]
"""

import sys
import json
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.common import record_text, render_mc
from goodwoolf.utils.data import JsonDataset, prepare_json_splits, split_files
from goodwoolf.utils.system import dataset_path

DATASET_NAME = "allenai/ai2_arc"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# Raw parquet shards (the HF export) live under ``raw/<config>/<split>``.
SOURCE_DIRNAME = "raw"
PARQUET_REVISION = "refs/convert/parquet"

# Both subsets, as ``(config, split)`` pairs.
SOURCES = (
    ("ARC-Easy", "train"),
    ("ARC-Challenge", "train"),
    ("ARC-Easy", "validation"),
    ("ARC-Challenge", "validation"),
    ("ARC-Easy", "test"),
    ("ARC-Challenge", "test"),
)

# Output split order; each is written to its own ``<split>.parquet``.
SPLITS = ("train", "validation", "test")


def fetch():
    """Download the ARC parquet shards via the HuggingFace Hub API."""
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
    """Build one GoodWoolf conversation record from an ARC row."""
    question = row["question"]
    letters = row["choices"]["label"]
    choices = row["choices"]["text"]
    answer = row["answerKey"]
    return {
        "messages": [
            {"role": "user", "content": render_mc(question, letters, choices)},
            {"role": "assistant", "content": answer},
        ],
        "answer": answer,
        "letters": letters,
        "source": "ai2_arc",
    }


def prepare():
    """Write one ``<split>.parquet`` per fetched ARC split."""
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
        total_chars = 0
        for record in dataset:
            count += 1
            total_chars += len(record_text(record))

        print(f"File:            {name}.parquet")
        print(f"Examples:        {count:,}")
        print(f"Total chars:     {total_chars:,}")
        if count:
            print(f"Avg chars/ex:    {total_chars / count:,.1f}")
        print(f"Format:          GoodWoolf conversation messages (SFT categorical)")


def sample(start=0, count=1, split=None):
    """Print COUNT samples starting at START, for one split or for every split."""
    for name in SPLITS if split is None else (split,):
        records = islice(load_data(name), start, start + count)
        print(f"Sample ({name}):")
        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Prepare ARC in GoodWoolf conversation format")
    actions = parser.add_subparsers(dest="action")

    actions.add_parser("fetch", help="Download the source dataset.")
    actions.add_parser("prepare", help="Write the prepared dataset.")

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
