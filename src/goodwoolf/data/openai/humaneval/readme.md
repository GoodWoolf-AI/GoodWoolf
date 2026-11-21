
# HumanEval dataset

[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) is a Python
coding benchmark of 164 hand-written programming problems. Each problem provides
a function signature/prompt, a canonical solution, and a set of unit tests used
to check a completion.

The dataset is prepared once for SFT (and evaluation). The records are
tokenizer-independent; the model tokenizer is applied at training time, when it
renders the conversation into tokens and builds the assistant loss mask.

Like the pretraining datasets, HumanEval is tokenizer-independent. It is stored
as JSONL in the message format used by nanochat (`nanochat/tasks/humaneval.py`).

## Prepare the data

```bash
python -m goodwoolf.data.openai.humaneval.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset openai/humaneval --fetch --prepare

# inspect some samples (all splits, or one with --split)
goodwoolf dataset openai/humaneval --sample 0 3
goodwoolf dataset openai/humaneval --sample 0 3 --split test
```

`--fetch` downloads the raw parquet export of the hub dataset via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/openai/humaneval/raw/openai_humaneval/test/`; `--prepare` reads those
local shards and writes the split file (`test.jsonl`) to:

```
~/.goodwoolf/data/openai/humaneval/
```

Each line is one problem:

```json
{"messages": [
  {"role": "user", "content": "def f(x):\n    \"\"\""},
  {"role": "assistant", "content": "def f(x):\n    \"\"\"\n    return x + 1\n"}
 ],
 "entry_point": "f",
 "test": "def check(candidate): ...",
 "source": "humaneval"}
```

`messages` mirrors nanochat's conversation format: the user message is the
prompt and the assistant message is the complete solution. `entry_point` and
`test` are kept so a completion can be executed and checked at evaluation time.

## Splits

Preparation preserves the fetched split: it is written to `test.jsonl`.
`load_data(split)` loads a single split and `load_data()` loads every split
combined.
