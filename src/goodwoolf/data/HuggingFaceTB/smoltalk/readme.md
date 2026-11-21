
# SmolTalk dataset

[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) is a
general conversational dataset used by SFT as the "general chat" part of the
training mixture. Each example is a conversation made of an optional system
message followed by alternating user/assistant turns.

The dataset is prepared once for SFT. The records are tokenizer-independent; the
model tokenizer is applied at training time, when it renders the conversation
into tokens and builds the assistant loss mask.

Like the pretraining datasets, SmolTalk is tokenizer-independent. It is stored
as JSONL in the message format used by nanochat (`nanochat/tasks/smoltalk.py`).

## Prepare the data

```bash
python -m goodwoolf.data.HuggingFaceTB.smoltalk.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset HuggingFaceTB/smoltalk --fetch --prepare

# inspect some samples (all splits, or one with --split)
goodwoolf dataset HuggingFaceTB/smoltalk --sample 0 3
goodwoolf dataset HuggingFaceTB/smoltalk --sample 0 3 --split test
```

`--fetch` downloads the raw parquet export of the hub dataset via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/HuggingFaceTB/smoltalk/raw/default/{train,test}/`; `--prepare` reads those
local shards and writes one `<split>.jsonl` per split (`train.jsonl`,
`test.jsonl`) to:

```
~/.goodwoolf/data/HuggingFaceTB/smoltalk/
```

Each line is one conversation:

```json
{"messages": [
  {"role": "user", "content": "hi"},
  {"role": "assistant", "content": "hello"}
 ],
 "source": "smoltalk"}
```

`messages` mirrors nanochat's conversation format, so nanochat's
`render_conversation` produces the same special-token structure and loss mask
during SFT.

## Splits

Preparation preserves the official splits: each is written to its own file
(`train.jsonl`, `test.jsonl`). `load_data(split)` loads a single split and
`load_data()` loads every split combined.
