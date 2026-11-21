# climbmix dataset

[ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) is a
400-billion-token pre-training dataset released by NVIDIA. It ships GPT-2 token
ids, so `prepare` detokenizes them back to text. The output is stored as
tokenizer-independent newline-delimited text files (`train.txt` and
`test.txt`); the last source file becomes `test.txt` and all earlier files form
`train.txt`. Tokenization happens at training time with the model tokenizer.

## Prepare the data

```bash
python -m goodwoolf.data.nvidia.climbmix.dataset --fetch --prepare
# or through the CLI
goodwoolf dataset nvidia/climbmix --fetch --prepare

# inspect some samples
goodwoolf dataset nvidia/climbmix --sample 0 3
```

`--fetch` downloads the tokenized JSONL shards (`part_*.tokenized.jsonl`) of
`nvidia/Nemotron-ClimbMix` via the HuggingFace Hub API (`snapshot_download`)
into `~/.goodwoolf/data/nvidia/climbmix/raw/`; `--prepare` detokenizes them with
tiktoken's `gpt2` encoding and writes `train.txt` and `test.txt` to:

```
~/.goodwoolf/data/nvidia/climbmix/
```

references:

- [CLIMB paper](https://arxiv.org/abs/2504.13161)
- [ClimbMix on the Hub](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
