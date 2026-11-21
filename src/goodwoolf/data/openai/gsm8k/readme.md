
# GSM8K dataset

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Grade School Math 8K) is a
set of ~8.5K grade-school math word problems. Each problem has a worked solution
that ends with the final numeric answer after a `####` marker, and many
solutions contain calculator annotations such as `<<12/60=0.2>>`.

The dataset is prepared once for both SFT and RL. The records are
tokenizer-independent; the model tokenizer is applied at training time, when it
renders the conversation into tokens and builds the assistant loss mask.

Like the pretraining datasets, GSM8K is tokenizer-independent. It is stored as
JSONL in the message/tool-call format used by nanochat
(`nanochat/tasks/gsm8k.py`).

## Prepare the data

```bash
python -m goodwoolf.data.openai.gsm8k.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset openai/gsm8k --fetch --prepare

# inspect some samples (all splits, or one with --split)
goodwoolf dataset openai/gsm8k --sample 0 3
goodwoolf dataset openai/gsm8k --sample 0 3 --split test
```

`--fetch` downloads the raw parquet export of the hub dataset via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/openai/gsm8k/raw/main/{train,test}/`; `--prepare` reads those local
shards and writes one `<split>.jsonl` per split (`train.jsonl`, `test.jsonl`) to:

```
~/.goodwoolf/data/openai/gsm8k/
```

Each line is one problem:

```json
{"messages": [
  {"role": "user", "content": "Weng earns $12 an hour ..."},
  {"role": "assistant", "content": [
    {"type": "text", "text": "Weng earns 12/60 = $"},
    {"type": "python", "text": "12/60"},
    {"type": "python_output", "text": "0.2"},
    {"type": "text", "text": " per minute.\n...\n#### 10"}
  ]}
 ],
 "question": "Weng earns $12 an hour ...",
 "answer": "10",
 "source": "gsm8k"}
```

`messages` mirrors nanochat's conversation format, so nanochat's
`render_conversation` produces the same special-token structure and loss mask
during SFT. `python` / `python_output` parts are the calculator tool calls
parsed from `<<expr=result>>`.

The same record is used for RL: `render_for_completion` keeps the user/prompt
part, and `answer` (the number after `####`) is the target for the verifiable
reward, matching nanochat's `GSM8K.reward`.

## Splits

Preparation preserves the official splits: each is written to its own file
(`train.jsonl`, `test.jsonl`). `load_data(split)` loads a single split (e.g.
`load_data("train")`) and `load_data()` loads every split combined.
