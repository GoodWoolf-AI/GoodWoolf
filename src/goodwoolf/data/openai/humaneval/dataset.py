"""Prepare HumanEval for training in nanochat conversation format.

HumanEval is a Python coding benchmark. It is prepared once and used by SFT
(and evaluation). The records are tokenizer-independent; the model tokenizer is
applied at training time.

Each output line is one problem::

    {"messages": [{"role": "user", "content": <prompt>},
                  {"role": "assistant", "content": <prompt + solution>}],
     "entry_point": <function name>,
     "test": <test cases>,
     "source": "humaneval"}

``messages`` mirrors ``nanochat/tasks/humaneval.py``: the user message is the
function signature/prefix and the assistant message is the complete solution.
``entry_point`` and ``test`` are kept so the completion can be executed and
checked at evaluation time.

The fetched split is preserved: it is written to ``test.parquet``.
``load_data(split)`` loads a single split and ``load_data()`` loads every split
combined.

Usage:
    python -m goodwoolf.data.openai.humaneval.dataset [--fetch] [--prepare] [--split SPLIT] [--sample START COUNT]
"""

import sys
import json
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.common import record_text
from goodwoolf.utils.data import JsonDataset, prepare_json_splits, split_files
from goodwoolf.utils.system import dataset_path

DATASET_NAME = "openai/openai_humaneval"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# Raw parquet shards (the HF export) live under ``raw/<config>/<split>``.
SOURCE_DIRNAME = "raw"
PARQUET_REVISION = "refs/convert/parquet"

# HumanEval only has a test split.
SOURCE = ("openai_humaneval", "test")
SOURCES = (SOURCE,)

# Output split order; each is written to its own ``<split>.parquet``.
SPLITS = ("test",)


def fetch():
    """Download the HumanEval parquet shards via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    config, split = SOURCE
    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        allow_patterns=[f"{config}/{split}/*.parquet"],
        local_dir=dest,
    )
    paths = sorted((dest / config / split).glob("*.parquet"))
    print(f"  {config}/{split}: {len(paths)} shard(s)", file=sys.stderr, flush=True)


def _conversation(row):
    """Build one nanochat conversation record from a HumanEval row."""
    prompt = row["prompt"]
    solution = row["canonical_solution"]
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{prompt}\n{solution}"},
        ],
        "entry_point": row["entry_point"],
        "test": row["test"],
        "source": "humaneval",
    }


def prepare():
    """Write one ``<split>.parquet`` per fetched HumanEval split."""
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
        print(f"Problems:        {count:,}")
        print(f"Total chars:     {total_chars:,}")
        if count:
            print(f"Avg chars/prob:  {total_chars / count:,.1f}")
        print(f"Format:          nanochat conversation messages (SFT code, exec-checked)")


def sample(start=0, count=1, split=None):
    """Print COUNT samples starting at START, for one split or for every split."""
    for name in SPLITS if split is None else (split,):
        records = islice(load_data(name), start, start + count)
        print(f"Sample ({name}):")
        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare HumanEval in nanochat conversation format")
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
