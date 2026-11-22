import argparse

from goodwoolf.utils.models import load_config, model_name_from_path
from goodwoolf.utils.training import build_token_bytes, save_token_bytes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build token byte counts for an HF model")
    args = parser.parse_args(argv)
    model_name = model_name_from_path(__file__)
    configs = load_config(model_name)
    tokenizer = configs["tokenizer"]
    if tokenizer is None:
        raise RuntimeError(f"No tokenizer configured for {model_name}")

    model_config = configs["model"]
    vocab_size = getattr(model_config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)

    token_bytes = build_token_bytes(tokenizer, vocab_size)
    save_token_bytes(model_name, token_bytes)
    print(f"Built {len(token_bytes):,} token byte counts for {model_name}", flush=True)


if __name__ == "__main__":
    main()
