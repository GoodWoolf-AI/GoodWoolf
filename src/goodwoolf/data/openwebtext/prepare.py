"""Tokenize OpenWebText with a local tokenizer and save as .bin files.

Usage:
    python -m goodwoolf.data.openwebtext.prepare --model-name Qwen/Qwen2-0.5B
"""

import os
import sys
import argparse
import numpy as np
from datasets import load_dataset
from goodwoolf.utils.models import load_config
from goodwoolf.utils.system import data_path

num_proc = 8
dataset_name = "Skylion007/openwebtext"


def fetch(model_name, data_name):
    config = load_config(model_name)
    enc = config["tokenizer"]
    enc.model_max_length = config["model"].max_position_embeddings
    eot = enc.eos_token_id

    data_dir = data_path(model_name, data_name)
    dataset = load_dataset(dataset_name, num_proc=num_proc)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)

    def process(example):
        ids = enc.encode(example['text'], add_special_tokens=False)
        ids.append(eot)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    os.makedirs(data_dir, exist_ok=True)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(data_dir, f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


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

    parser = argparse.ArgumentParser(description="Tokenize OpenWebText with a local tokenizer")
    parser.add_argument("--model-name", required=True, help="Model name (e.g. Qwen/Qwen2-0.5B)")
    args = parser.parse_args(argv)

    fetch(args.model_name, data_name)

    metadata(args.model_name, data_name, "train")
    sample(args.model_name, data_name, "train", 0, 50)

    metadata(args.model_name, data_name, "test")
    sample(args.model_name, data_name, "test", 0, 50)


if __name__ == '__main__':
    main()
