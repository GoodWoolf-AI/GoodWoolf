
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

Learn more by following the online documenation:

- [Get Started](https://docs.goodwoolf.ai/)
  - [Introduction](https://docs.goodwoolf.ai/introduction)
  - [Supported Datasets](#supported-datasets)
- [Features](https://docs.goodwoolf.ai/category/features)
  - [Training Process](https://docs.goodwoolf.ai/features/training-process)
  - [Training Ledger](https://docs.goodwoolf.ai/features/training-ledger)
  - [Collective Training Time](https://docs.goodwoolf.ai/features/training-time)
  - [Model Poisoning Controls](https://docs.goodwoolf.ai/features/model-poisoning)
- [Configure](https://docs.goodwoolf.ai/category/configure)
  - [Python Environment](https://docs.goodwoolf.ai/setup/environment)
  - [Virtaul Memory](https://docs.goodwoolf.ai/setup/virtual-memory)
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
=======================================================
Visual represenation of Collaborative Training Workflow
=======================================================

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

## Supported Datasets

You can train GoodWoolf on your own private data. To make things easier, a few
public datasets are also integrated so you can use them right away.

Datasets live in `src/goodwoolf/data/<vendor>/<dataset>/` and are prepared with
`goodwoolf dataset <vendor>/<dataset> --fetch --prepare`. The **Training stage** column marks
the stage each dataset is intended for: `NTP` (base training with next token
prediction), `SFT` (supervised fine-tuning), or `RL` (reinforcement learning).

| Dataset | Description | Training stage |
| --- | --- | --- |
| `nvidia/climbmix` | ClimbMix-400B pretraining corpus (GPT-2 tokenized). | NTP |
| `karpathy/climbmix-400b-shuffle` | Nanochat-style globally shuffled ClimbMix-400B raw-text corpus. | NTP |
| `Skylion007/openwebtext` | OpenWebText web-text corpus for pretraining. | NTP |
| `Salesforce/wikitext` | WikiText-2 language-modeling corpus. | NTP |
| `allenai/ai2_arc` | AI2 Reasoning Challenge, multiple-choice science questions. | SFT |
| `openai/humaneval` | HumanEval, Python coding problems with unit tests. | SFT |
| `cais/mmlu` | MMLU, multiple-choice knowledge across 57 subjects. | SFT |
| `HuggingFaceTB/smoltalk` | SmolTalk, general conversational data. | SFT |
| `openai/gsm8k` | GSM8K, grade-school math word problems with worked solutions. | SFT, RL |

## Usage Examples

### Prepare training data

```bash
goodwoolf dataset <dataset> --fetch --prepare
```

### Download model weights

```bash
goodwoolf pull <vendor>/<model>
```

### Train model

```bash
goodwoolf train <vendor>/<model> --stage NTP --data-name <dataset>
```

`--stage` selects the training stage: `TOK` (tokenizer training; HF templates
only prepare the token-byte cache), `NTP` (base training with next token
prediction), `SFT` (supervised fine-tuning, full answer prediction), or `RL`
(reinforcement, free thinking with final answer prediction).

### Generate text

```bash
goodwoolf run <vendor>/<model> --prompt "A long time ago..."
```

## Get involved

GoodWoolf is built by an open-source community project.

Whether you're trainig and sharing models, improving the platform, fixing bugs or writing documentation, your contributions are welcome.

## License
- [GoodWoolf](https://github.com/GoodWoolf-AI/GoodWoolf): MIT
- [GoodWoolf Training Ledger](https://hf.co/datasets/GoodWoolf/Training-Ledger): OpenMDW