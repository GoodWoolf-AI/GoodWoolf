"""Prepare GSM8K for training in nanochat conversation format.

GSM8K is prepared once and used by both SFT and RL. The records are
tokenizer-independent; the model tokenizer is applied at training time, when
nanochat renders the conversation (roles, tool calls and the assistant loss
mask) and when RL extracts the verifiable ``#### <answer>`` reward.

Each output line is one problem::

    {"messages": [{"role": "user", "content": <question>},
                  {"role": "assistant", "content": [{"type": "text", ...},
                                                    {"type": "python", ...},
                                                    {"type": "python_output", ...}]}],
     "question": <question>,
     "answer": <final numeric answer, e.g. "72">,
     "source": "gsm8k"

``messages`` mirrors ``nanochat/tasks/gsm8k.py``: calculator annotations
(``<<expr=result>>``) become tool-call parts, so the tokenizer can render the
same ``<|python_start|>`` / ``<|output_start|>`` structure used by nanochat
during SFT and RL. ``answer`` is the number after ``####`` used by the
verifiable RL reward.

The fetched splits are preserved: each is written to its own ``<split>.parquet``
(``train.parquet``, ``test.parquet``). ``load_data(split)`` loads a single split and
``load_data()`` loads every split combined.

Usage:
    python -m goodwoolf.data.openai.gsm8k.dataset [--fetch] [--prepare] [--sample START COUNT]
"""

import sys
import re
import json
import argparse
from itertools import islice
from pathlib import Path

from goodwoolf.utils.common import record_text
from goodwoolf.utils.data import JsonDataset, prepare_json_splits, split_files
from goodwoolf.utils.system import dataset_path

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATA_NAME = "/".join(Path(__file__).parent.parts[-2:])

# Raw parquet shards (the HF export) live under ``raw/<config>/<split>``.
SOURCE_DIRNAME = "raw"
SPLITS = ("train", "test")
SOURCES = tuple((DATASET_CONFIG, split) for split in SPLITS)

# Every Hub dataset has an auto-generated parquet export on this revision.
PARQUET_REVISION = "refs/convert/parquet"

# Same answer extraction as nanochat: the number after the ``####`` marker.
ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")

# Same calculator annotations as nanochat.
CALC_RE = re.compile(r"(<<[^>]+>>)")


def fetch():
    """Download the GSM8K parquet shards (train and test) via the HuggingFace Hub API."""
    from huggingface_hub import snapshot_download

    dest = Path(dataset_path(DATA_NAME)) / SOURCE_DIRNAME
    snapshot_download(
        DATASET_NAME,
        repo_type="dataset",
        revision=PARQUET_REVISION,
        allow_patterns=[f"{DATASET_CONFIG}/{split}/*.parquet" for split in SPLITS],
        local_dir=dest,
    )
    for split in SPLITS:
        paths = sorted((dest / DATASET_CONFIG / split).glob("*.parquet"))
        print(f"  {split}: {len(paths)} shard(s) -> {dest / DATASET_CONFIG / split}", file=sys.stderr, flush=True)


def prepare():
    """Write one ``<split>.parquet`` per fetched GSM8K split."""
    prepare_json_splits(dataset_path(DATA_NAME), SOURCES, SPLITS, SOURCE_DIRNAME)


def _conversation(row):
    """Build one nanochat conversation record from a GSM8K row."""
    question = row["question"].strip()
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": _split_solution(row["answer"])},
        ],
        "question": question,
        "answer": _final_answer(row["answer"]),
        "source": "gsm8k",
    }


def _final_answer(answer):
    """Return the numeric answer after ``####`` (or ``None`` if missing)."""
    match = ANSWER_RE.search(answer)
    return match.group(1).replace(",", "") if match else None


def _split_solution(answer):
    """Split GSM8K calculator annotations (``<<expr=result>>``) into nanochat
    assistant content parts.

    The GSM8K dataset embeds calculator expressions as ``<<expr=result>>`` markers.
    This function parses those markers and returns a list of part dictionaries
    matching the nanochat format::

        [{"type": "python",   "text": "expr"},
         {"type": "python_output", "text": "result"}]

    Plain text between markers is returned as ``{"type": "text", "text": "..."}``.

    Parameters
    ----------
    answer : str
        A GSM8K problem's answer string, possibly containing ``<<expr=result>>`` markers.

    Returns
    -------
    list[dict]
        A list of part dictionaries, each with ``"type"`` and ``"text"`` keys.
        Types are ``"python"``, ``"python_output"``, or ``"text"``.
    """
    parts = []
    for part in CALC_RE.split(answer):
        if part.startswith("<<") and part.endswith(">>"):
            inner = part[2:-2]
            if "=" in inner:
                expr, result = inner.rsplit("=", 1)
            else:
                expr, result = inner, ""
            parts.append({"type": "python", "text": expr})
            parts.append({"type": "python_output", "text": result})
        else:
            parts.append({"type": "text", "text": part})
    return parts


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
        total_tools = 0
        missing_answers = 0
        for record in dataset:
            count += 1
            for message in record["messages"]:
                content = message["content"]
                if isinstance(content, list):
                    total_tools += sum(part["type"] == "python" for part in content)
            total_chars += len(record_text(record))
            if record["answer"] is None:
                missing_answers += 1

        print(f"File:            {name}.parquet")
        print(f"Problems:        {count:,}")
        print(f"Tool calls:      {total_tools:,}")
        print(f"Missing answers: {missing_answers:,}")
        print(f"Total chars:     {total_chars:,}")
        if count:
            print(f"Avg chars/prob:  {total_chars / count:,.1f}")
        print(f"Format:          nanochat conversation messages (SFT + verifiable RL answer)")


def sample(start=0, count=1, split=None):
    """Print COUNT samples starting at START, for one split or for every split."""
    for name in SPLITS if split is None else (split,):
        records = islice(load_data(name), start, start + count)
        print(f"Sample ({name}):")
        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare GSM8K in nanochat conversation format")
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