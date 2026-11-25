
# AI2 ARC dataset

[ARC](https://huggingface.co/datasets/allenai/ai2_arc) (AI2 Reasoning Challenge)
is a multiple-choice science benchmark with two subsets, `ARC-Easy` and
`ARC-Challenge`. Each question has a few labeled answer choices and a single
correct `answerKey`.

The dataset is prepared once for SFT (and evaluation). The records are
tokenizer-independent; the model tokenizer is applied at training time, when it
renders the conversation into tokens and builds the assistant loss mask.

Like the pretraining datasets, ARC is tokenizer-independent. It is stored as
JSONL in GoodWoolf's message format.

## Prepare the data

```bash
python -m goodwoolf.data.allenai.ai2_arc.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset allenai/ai2_arc --fetch --prepare

# inspect some samples (all splits, or one with --split)
goodwoolf dataset allenai/ai2_arc --sample 0 3
goodwoolf dataset allenai/ai2_arc --sample 0 3 --split test
```

`--fetch` downloads the raw parquet export of the hub dataset via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/allenai/ai2_arc/raw/{ARC-Easy,ARC-Challenge}/{train,validation,test}/`;
`--prepare` reads those local shards and writes one `<split>.jsonl` per split
(`train.jsonl`, `validation.jsonl`, `test.jsonl`) to:

```
~/.goodwoolf/data/allenai/ai2_arc/
```

Each line is one example:

```json
{"messages": [
  {"role": "user", "content": "Multiple Choice question: Which is a mammal?\n- Dog=A\n- Snake=B\n- Frog=C\n- Fish=D\n\nRespond only with the letter of the correct answer."},
  {"role": "assistant", "content": "A"}
 ],
 "answer": "A",
 "letters": ["A", "B", "C", "D"],
 "source": "ai2_arc"}
```

`messages` uses GoodWoolf's conversation format: the user message renders the
question with its choices via `render_mc` and the assistant message is the
answer letter. `letters` is kept so the answer can be validated.

## Splits

Preparation preserves the official splits (both subsets combined per split):
each is written to its own file (`train.jsonl`, `validation.jsonl`,
`test.jsonl`). `load_data(split)` loads a single split and `load_data()` loads
every split combined.
