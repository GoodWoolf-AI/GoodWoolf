# kl3m-usc dataset

The data is stored as tokenizer-independent parquet text (`train.parquet` and
`test.parquet`), with one document per row in the `text` column. Tokenization
happens at training time with the model tokenizer.

The source dataset, `alea-institute/kl3m-data-usc`, stores KL3M token ids. During
preparation those ids are decoded with the `alea-institute/kl3m-004-128k-cased`
tokenizer. The single source shard is split by document count: the first 80%
becomes `train.parquet` and the final 20% becomes `test.parquet`.

## Prepare the data

```bash
goodwoolf dataset alea-institute/kl3m-usc --fetch --prepare

# inspect some samples
goodwoolf dataset alea-institute/kl3m-usc --sample 0 3
```

`--fetch` downloads the source parquet into
`~/.goodwoolf/data/alea-institute/kl3m-usc/raw/data/`; `--prepare` decodes it and writes:

```
~/.goodwoolf/data/alea-institute/kl3m-usc/
```

The source dataset is
[`alea-institute/kl3m-data-usc`](https://huggingface.co/datasets/alea-institute/kl3m-data-usc).
