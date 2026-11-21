"""Train the Puppy tokenizer and save it with the model weights."""

import argparse
from pathlib import Path

import torch
import yaml

from goodwoolf.utils.data import load_data
from goodwoolf.utils.models import config_path, model_name_from_path
from goodwoolf.utils.system import import_file, tokenizer_path


def main(argv=None):
    model_name = model_name_from_path(__file__)
    tokenizer_module = import_file(
        str(Path(__file__).with_name("tokenizer.py")),
        f"tokenizer_{model_name.replace('/', '_')}",
    )
    Tokenizer = tokenizer_module.Tokenizer

    with open(config_path(model_name) / "model.yaml") as f:
        tok_config = yaml.safe_load(f).get("train_tok", {})
    parser = argparse.ArgumentParser(description="Train the Puppy tokenizer")
    parser.add_argument(
        "--data-name",
        default=tok_config.get("data_name", "nvidia/climbmix"),
        help="Dataset name (e.g. nvidia/climbmix)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to train on")
    parser.add_argument("--max-chars", type=int, default=tok_config.get("max_chars", 2_000_000_000))
    parser.add_argument("--doc-cap", type=int, default=tok_config.get("doc_cap", 10_000))
    parser.add_argument("--vocab-size", type=int, default=tok_config.get("vocab_size", 32768))
    args = parser.parse_args(argv)
    args.model_name = model_name

    dataset = load_data(args.data_name, split=args.split)

    def text_iterator():
        nchars = 0
        for doc in dataset:
            doc_text = doc[:args.doc_cap] if len(doc) > args.doc_cap else doc
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return

    print(f"Training tokenizer on {args.data_name}/{args.split}", flush=True)
    tokenizer = Tokenizer.train_from_iterator(text_iterator(), args.vocab_size)

    path = Path(tokenizer_path(model_name))
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(path)
    print(f"Saved tokenizer to {path / 'tokenizer.pkl'}", flush=True)

    special_ids = set(tokenizer.encode_special(token) for token in tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            token_bytes.append(len(tokenizer.decode_single_token_bytes(token_id)))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), path / "token_bytes.pt")
    print(f"Saved token bytes to {path / 'token_bytes.pt'}", flush=True)

    yaml_path = config_path(model_name) / "model.yaml"
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    config["vocab_size"] = tokenizer.get_vocab_size()
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["pad_token_id"] = tokenizer.bos_token_id
    with open(yaml_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Updated {yaml_path}", flush=True)


if __name__ == "__main__":
    main()
