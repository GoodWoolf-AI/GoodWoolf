"""Puppy tokenizer, trained with rustbpe and used with tiktoken."""

import os
import pickle
from functools import lru_cache
from pathlib import Path

from goodwoolf.utils.system import tokenizer_path

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, model_name):
        path = Path(tokenizer_path(model_name)) / "tokenizer.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {path}; run with --stage TOK first"
            )
        with open(path, "rb") as f:
            self.enc = pickle.load(f)
        self.bos_token_id = self.enc.encode_single_token("<|bos|>")
        self.eos_token_id = self.bos_token_id
        self.pad_token_id = self.bos_token_id

    @classmethod
    def from_encoding(cls, enc):
        tokenizer = cls.__new__(cls)
        tokenizer.enc = enc
        tokenizer.bos_token_id = tokenizer.enc.encode_single_token("<|bos|>")
        tokenizer.eos_token_id = tokenizer.bos_token_id
        tokenizer.pad_token_id = tokenizer.bos_token_id
        return tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        import rustbpe
        import tiktoken

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls.from_encoding(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls.from_encoding(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, token_id):
        return self.enc.decode([token_id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def render_conversation(self, conversation, max_tokens=2048):
        if conversation["messages"][0]["role"] == "system":
            system = conversation["messages"][0]["content"]
            user = conversation["messages"][1]["content"]
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}] + conversation["messages"][2:]
        else:
            messages = conversation["messages"]

        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            role = message["role"]
            assert role == ("user" if i % 2 == 0 else "assistant"), f"Message {i} is from {role}"
            content = message["content"]
            if role == "user":
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
                continue

            add_tokens(assistant_start, 0)
            if isinstance(content, str):
                add_tokens(self.encode(content), 1)
            else:
                for part in content:
                    part_ids = self.encode(part["text"])
                    if part["type"] == "text":
                        add_tokens(part_ids, 1)
                    elif part["type"] == "python":
                        add_tokens(python_start, 1)
                        add_tokens(part_ids, 1)
                        add_tokens(python_end, 1)
                    elif part["type"] == "python_output":
                        add_tokens(output_start, 0)
                        add_tokens(part_ids, 0)
                        add_tokens(output_end, 0)
                    else:
                        raise ValueError(f"Unknown part type: {part['type']}")
            add_tokens(assistant_end, 1)

        return ids[:max_tokens], mask[:max_tokens]

    def render_prompt(self, messages, max_tokens=2048):
        ids, _ = self.render_conversation(
            {"messages": list(messages) + [{"role": "assistant", "content": ""}]},
            max_tokens=max_tokens,
        )
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        if ids and ids[-1] == assistant_end:
            ids.pop()
        if len(ids) >= max_tokens:
            ids = ids[:-1]
        if not ids or ids[-1] != assistant_start:
            ids.append(assistant_start)
        return ids


    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            special_ids = {self.encode_special(token) for token in self.get_special_tokens()}
            ids = [token_id for token_id in ids if token_id not in special_ids]
        return self.enc.decode(list(ids))

    def decode_single_token_bytes(self, token_id):
        return self.enc.decode_single_token_bytes(token_id)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(self.enc, f)
