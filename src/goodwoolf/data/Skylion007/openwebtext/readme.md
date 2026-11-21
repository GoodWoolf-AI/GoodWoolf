# openwebtext dataset

The data is stored as tokenizer-independent newline-delimited text files
(`train.txt` and `test.txt`). Preparation splits the source shards by file: the
last source shard becomes `test.txt` and all earlier shards form `train.txt`.
Tokenization happens at training time with the model tokenizer.

## Prepare the data

```bash
python -m goodwoolf.data.Skylion007.openwebtext.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset Skylion007/openwebtext --fetch --prepare

# inspect some samples
goodwoolf dataset Skylion007/openwebtext --sample 0 3
```

`--fetch` downloads the raw parquet export of `Skylion007/openwebtext` via the
HuggingFace Hub API (`snapshot_download` of `refs/convert/parquet`) into
`~/.goodwoolf/data/Skylion007/openwebtext/raw/plain_text/train/`; `--prepare` reads those
local shards and writes `train.txt` and `test.txt` to:

```
~/.goodwoolf/data/Skylion007/openwebtext/
```

The source dataset is `Skylion007/openwebtext` (config `plain_text`).

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
