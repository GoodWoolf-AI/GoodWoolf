
<p align="center">
  <a href="https://goodwoolf.ai">
    <img src="https://github.com/GoodWoolf-AI/.github/raw/main/profile/goodwoolf.svg" width="200" alt="GoodWoolf" />
  </a>
</p>

<h1 align="center">
  <a href="https://goodwoolf.ai">GoodWoolf</a>
</h1>

<p align="center">
  <strong>AI by the People</strong>
</p>

<p align="center">
  Foundation Models · Commodity Hardware · Collaborative Training
</p>

<p align="center">
  <a href="https://goodwoolf.ai">Website</a>
  &nbsp;·&nbsp;
  <a href="https://docs.goodwoolf.ai">Docs</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/GoodWoolf-AI/GoodWoolf">Main Repository</a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/GoodWoolf/Training-Ledger">Training-Ledger</a>
</p>

<br/>

## Building Foundation Models Collaboratively on Commodity Hardware

GoodWoolf builds large foundation models on commodity hardware using the following principles:

- **Open Collaboration:** Enable collective model development across independent contributors, coordinated through a shared public training ledger.

- **Virtual Memory Extension:** Expanding effective memory beyond physical RAM by leveraging storage-based virtual memory.

- **Data-Independent Scaling:** Decoupling total training time from global data volume, bounding system latency to individual run durations rather than aggregate data sizes.

Learn more by following the online documentation:

- [Get Started](https://docs.goodwoolf.ai/)
  - [Introduction](https://docs.goodwoolf.ai/introduction)
  - [Supported Models](#supported-models)
  - [Supported Datasets](#supported-datasets)
- [Features](https://docs.goodwoolf.ai/category/features)
  - [Training Process](https://docs.goodwoolf.ai/features/training-process)
  - [Training Ledger](https://docs.goodwoolf.ai/features/training-ledger)
  - [Collective Training Time](https://docs.goodwoolf.ai/features/training-time)
  - [Model Poisoning Controls](https://docs.goodwoolf.ai/features/model-poisoning)
- [Configure](https://docs.goodwoolf.ai/category/configure)
  - [Python Environment](https://docs.goodwoolf.ai/setup/environment)
  - [Virtual Memory](https://docs.goodwoolf.ai/setup/virtual-memory)
- [Usage](https://docs.goodwoolf.ai/category/usage)
  - [Naming Conventions](https://docs.goodwoolf.ai/usage/naming-conventions)
  - [Prepare training data](https://docs.goodwoolf.ai/usage/prepare-data)
  - [Download model weights](https://docs.goodwoolf.ai/usage/download-model)
  - [Train model](https://docs.goodwoolf.ai/usage/train-model)
  - [Generate text](https://docs.goodwoolf.ai/usage/generate-text)
  - [Create a new model](https://docs.goodwoolf.ai/usage/create-model)
  - [Publish a new model](https://docs.goodwoolf.ai/usage/publish-model)
  - [Publish a model update](https://docs.goodwoolf.ai/usage/publish-update)

## Project Structure

The project consists of two repositories:

- [GoodWoolf](https://github.com/GoodWoolf-AI/GoodWoolf) (this repository) - the training framework
- [GoodWoolf Training Ledger](https://hf.co/datasets/GoodWoolf/Training-Ledger) - the public training log

### Collaborative Training Workflow

The training process is hardware-agnostic, parallel, and completely asynchronous.

```bash
========================================================
Visual representation of Collaborative Training Workflow
========================================================

          +-----------+----------+-----------+
          |           |          |           |
          v           |          |           |
  +----------------+  |          v           |
  | Participant A  |  |  +----------------+  |
  |----------------|  |  | Participant B  |  |
  | Local data     |  |  |----------------|  |
  | Train locally  |  |  | Local data     |  |
  | Publish model  |  |  | Train locally  |  |
  | Pull request   |  |  | Publish model  |  |
  +-------+--------+  |  | Pull request   |  |
          |           v  +--------+-------+  |
          |   +----------------+  |          |
          |   | Participant C  |  |          |
          |   |----------------|  |          |
          |   | Local data     |  |          |
          |   | Train locally  |  |          |
          |   | Model offload  |  |          |
          |   | Pull request   |  |          |
          |   +------+---------+  |          |
          |          |            |          |
          v          v            v          |
       +-----------------------------+       |
       |   PUBLIC TRAINING LEDGER    |-------+
       |-----------------------------|
       | Base model weights          |
       | Training updates / PRs      |<------+
       | Training history            |       |
       +-----------------------------+       |
                     |                       |
                     v                       |
       +-----------------------------+       |
       |   Peers / Independent       |       |
       |   Auditors                  |-------+
       |-----------------------------|
       | Validate and aggregate      |
       | model updates               |
       | Publish updated model       |
       | Pull request                |
       +-----------------------------+
```

## Supported Models

Models live in `src/goodwoolf/models/<vendor>/<model>/`. Each bundled model
includes runtime code and training entry points for `TOK`, `NTP`, `SFT`, and
`RL`.

| Model | Architecture | Notes |
| --- | --- | --- |
| `GoodWoolf/Puppy` | Custom LLaMA-style transformer; 12 layers, 768 hidden size, 32K vocab, 2048 context. | Native GoodWoolf model; `TOK` trains a RustBPE tokenizer. |
| `OpenAI-Community/GPT2` | GPT-2 small; 12 layers, 768 hidden size, 50,257 vocab, 1024 context. | Hugging Face GPT-2 wrapper. `TOK` prepares the token-byte cache instead of training a new tokenizer. |
| `Qwen/Qwen2-0.5B` | Qwen2 0.5B; 24 layers, 896 hidden size, 151,936 vocab, 131,072 context. | Hugging Face Qwen2 wrapper. `TOK` prepares the token-byte cache instead of training a new tokenizer. |

Run `gw model --list` or `gw model --list <vendor>` to list local models. Use
`gw model --list --remote hf [vendor]` to list Hugging Face models.

## Supported Datasets

You can train GoodWoolf on your own private data. To make things easier, a few
public datasets are also integrated so you can use them right away.

Datasets live in `src/goodwoolf/data/<vendor>/<dataset>/` and are prepared with
`gw dataset <vendor>/<dataset> --fetch --prepare`. Run `gw dataset --list` to see
the current list. The **Training stage** column marks the stage each dataset is
intended for: `NTP` (base training with next token prediction), `SFT`
(supervised fine-tuning), or `RL` (reinforcement learning).

| Dataset | Description | Training stage |
| --- | --- | --- |
| `nvidia/climbmix` | ClimbMix-400B pretraining corpus prepared as tokenizer-independent text. | NTP |
| `alea-institute/kl3m-usc` | KL3M US Code legal text corpus prepared as tokenizer-independent parquet text. | NTP |
| `Skylion007/openwebtext` | OpenWebText web-text corpus for pretraining. | NTP |
| `Salesforce/wikitext` | WikiText-2 language-modeling corpus. | NTP |
| `allenai/ai2_arc` | AI2 Reasoning Challenge, multiple-choice science questions. | SFT |
| `openai/humaneval` | HumanEval, Python coding problems with unit tests. | SFT |
| `cais/mmlu` | MMLU, multiple-choice knowledge across 57 subjects. | SFT |
| `HuggingFaceTB/smoltalk` | SmolTalk, general conversational data. | SFT |
| `openai/gsm8k` | GSM8K, grade-school math word problems with worked solutions. | SFT, RL |

## Usage Examples

`goodwoolf` and `gw` are equivalent CLI names.

### Inspect available components

```bash
gw vendor --list
gw model --list
gw model --list GoodWoolf
gw dataset --list
```

### Prepare training data

```bash
gw dataset <vendor>/<dataset> --fetch --prepare
```

For example:

```bash
gw dataset nvidia/climbmix --fetch --prepare
gw dataset HuggingFaceTB/smoltalk --fetch --prepare
```

### Fetch or create a model

```bash
gw pull <vendor>/<model> [--model-id <MODEL_ID>]
gw model <vendor>/<model> --config --weights
gw model <new-vendor>/<new-model> --template OpenAI-Community/GPT2
```

### Train model

```bash
gw train <vendor>/<model> --stage NTP --data-name <vendor>/<dataset>
```

For example:

```bash
gw train GoodWoolf/Puppy --stage TOK --data-name nvidia/climbmix
gw train GoodWoolf/Puppy --stage NTP --data-name nvidia/climbmix
gw train GoodWoolf/Puppy --stage SFT --data-name HuggingFaceTB/smoltalk
gw train GoodWoolf/Puppy --stage RL --data-name openai/gsm8k
```

`--stage` selects the training stage: `TOK` (tokenizer training; HF models
prepare the token-byte cache), `NTP` (base training with next token
prediction), `SFT` (supervised fine-tuning, full answer prediction), or `RL`
(reinforcement, free thinking with final answer prediction).

### Generate text

```bash
gw run <vendor>/<model> --prompt "A long time ago..." --max-tokens 100
```

### Chat interactively

```bash
gw chat <vendor>/<model> --max-tokens 256 --temperature 0.6 --top-k 50
```

`gw chat` loads the active weights for the model name. Type `quit` or `exit`
to end the session, or `clear` to start a new conversation. For a single
prompt, use `gw run`.

### Save and manage checkpoints

```bash
gw save <vendor>/<model> [--load-path <checkpoint>]
gw checkout <vendor>/<model> --model-id <MODEL_ID>
gw merge <vendor>/<model> --data-name <vendor>/<dataset> --model-id <MODEL_ID>
gw ledger <vendor>/<model> --check
```

## Get involved

The idea behind GoodWoolf is to build large models together as an open-source community.
Building together is cheaper, faster, and more environmentally friendly.

Whether you're training and sharing models, improving the platform, fixing bugs, or writing documentation, your contributions are welcome.

## License
- [GoodWoolf](https://github.com/GoodWoolf-AI/GoodWoolf): MIT
- [GoodWoolf Training Ledger](https://hf.co/datasets/GoodWoolf/Training-Ledger): OpenMDW