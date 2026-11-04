"""Tokenize Shakespeare with a local tokenizer and save as .bin files.

Usage:
    python -m goodwoolf.data.shakespeare.prepare --model-name SupraLabs/Supra-Mini-0.1M
"""

import os
import sys
import argparse
import requests
import numpy as np

from goodwoolf.utils.models import load_config
from goodwoolf.utils.system import data_path

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def fetch(model_name, data_name):
    config = load_config(model_name)
    enc = config["tokenizer"]
    enc.model_max_length = config["model"].max_position_embeddings
    eot = enc.eos_token_id

    data_dir = data_path(model_name, data_name)
    os.makedirs(data_dir, exist_ok=True)

    input_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(input_path):
        print(f"Downloading {data_url} ...", file=sys.stderr, flush=True)
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

    print(f"Loading input.txt ...", file=sys.stderr, flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    n = len(data)

    train_data = data[: int(n * 0.9)]
    test_data = data[int(n * 0.9) :]

    for split_name, text_data in [("train", train_data), ("test", test_data)]:
        print(f"Tokenizing {split_name} ({len(text_data):,} chars) ...", file=sys.stderr, flush=True)
        ids = enc.encode(text_data, add_special_tokens=False)
        ids.append(eot)
        print(f"  {len(ids):,} tokens", file=sys.stderr, flush=True)

        filename = os.path.join(data_dir, f"{split_name}.bin")
        np.array(ids, dtype=np.uint16).tofile(filename)
        print(f"  Saved to {filename}", file=sys.stderr, flush=True)


def load_data(model_name: str, data_name: str, split: str) -> np.ndarray:
    path = os.path.join(data_path(model_name, data_name), f'{split}.bin')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return np.memmap(path, dtype=np.uint16, mode='r')


def metadata(model_name: str, data_name: str, split: str):
    config = load_config(model_name)
    enc = config["tokenizer"]
    tokens = load_data(model_name, data_name, split)
    total_tokens = len(tokens)
    total_docs = int(np.sum(tokens == enc.eos_token_id))

    print(f"File:           {split}.bin")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Total docs:     {total_docs:,}")
    if total_docs > 0:
        print(f"Avg tokens/doc: {total_tokens / total_docs:,.1f}")
    print(f"Token dtype:    {tokens.dtype}")
    print(f"EOT token id:   {enc.eos_token_id}")


def sample(model_name: str, data_name: str, split: str, start: int, count: int):
    config = load_config(model_name)
    enc = config["tokenizer"]
    tokens = load_data(model_name, data_name, split)
    end = min(start + count, len(tokens))
    slice_tokens = tokens[start:end].tolist()
    text = enc.decode(slice_tokens)

    print(f"{split.upper()} sample:")
    print(text)


def main(argv=None):
    data_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Tokenize Shakespeare with a local tokenizer")
    parser.add_argument("--model-name", required=True, help="Model name (e.g. SupraLabs/Supra-Mini-0.1M)")
    args = parser.parse_args(argv)

    fetch(args.model_name, data_name)

    metadata(args.model_name, data_name, "train")
    sample(args.model_name, data_name, "train", 0, 50)

    metadata(args.model_name, data_name, "test")
    sample(args.model_name, data_name, "test", 0, 50)


if __name__ == "__main__":
    main()
