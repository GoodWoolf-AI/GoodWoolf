
# MMLU dataset

[MMLU](https://huggingface.co/datasets/cais/mmlu) (Massive Multitask Language
Understanding) is a multiple-choice knowledge benchmark spanning 57 subjects.
Each question has four choices and a single correct answer index.

The dataset is prepared once for SFT (and evaluation). The records are
tokenizer-independent; the model tokenizer is applied at training time, when it
renders the conversation into tokens and builds the assistant loss mask.

Like the pretraining datasets, MMLU is tokenizer-independent. It is stored as
JSONL in GoodWoolf's message format.

## Prepare the data

```bash
python -m goodwoolf.data.cais.mmlu.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset cais/mmlu --fetch --prepare

# inspect some samples (all splits, or one with --split)
goodwoolf dataset cais/mmlu --sample 0 3
goodwoolf dataset cais/mmlu --sample 0 3 --split test
```

`--fetch` downloads the raw parquet export of the hub dataset via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/cais/mmlu/raw/all/{auxiliary_train,test}/`; `--prepare` reads those
local shards and writes one `<split>.jsonl` per split (`auxiliary_train.jsonl`,
`test.jsonl`) to:

```
~/.goodwoolf/data/cais/mmlu/
```

Each line is one example:

```json
{"messages": [
  {"role": "user", "content": "Multiple Choice question: Q1\n- a=A\n- b=B\n- c=C\n- d=D\n\nRespond only with the letter of the correct answer."},
  {"role": "assistant", "content": "B"}
 ],
 "answer": "B",
 "letters": ["A", "B", "C", "D"],
 "subject": "college_biology",
 "source": "mmlu"}
```

`messages` uses GoodWoolf's conversation format: the user message renders the
question with its four choices via `render_mc` and the assistant message is the
answer letter. `subject` and `letters` are kept for reference.

## Splits

Preparation preserves the fetched splits: each is written to its own file
(`auxiliary_train.jsonl`, `test.jsonl`). `load_data(split)` loads a single split
and `load_data()` loads every split combined.
