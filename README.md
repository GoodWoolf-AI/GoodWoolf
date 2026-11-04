<p align="center">
  <a href="https://goodwoolf.ai">
    <img src="img/goodwoolf.svg" alt="GoodWoolf" width="200"/>
  </a>
</p>

# GoodWoolf

GoodWoolf builds large foundation models on commodity hardware using the following principles:

- **Virtual Memory Extension:** Expanding effective memory beyond physical RAM by leveraging storage-based virtual memory.
- **Central Training Ledger:** Coordinating fully asynchronous, decentralized state transitions through a shared public ledger.
- **Data-Independent Scaling:** Decoupling total training time from global data volume, bounding system latency to individual run durations rather than aggregate data sizes.

Learn more by following the online documenation:

- [Get Started](https://docs.googwoolf.ai/)
  - [Introduction](https://docs.googwoolf.ai/introduction)
- [Features](https://docs.googwoolf.ai/category/features)
  - [Training Process](https://docs.googwoolf.ai/features/training-process)
  - [Training Ledger](https://docs.googwoolf.ai/features/training-ledger)
  - [Collective Training Time](https://docs.googwoolf.ai/features/training-time)
  - [Model Poisoning Controls](https://docs.googwoolf.ai/features/model-poisoning)
- [Configure](https://docs.googwoolf.ai/category/configure)
  - [Python Environment](https://docs.googwoolf.ai/setup/environment)
  - [Virtaul Memory](https://docs.googwoolf.ai/setup/virtual-memory)
- [Usage](https://docs.googwoolf.ai/category/usage)
  - [Naming Conventions](https://docs.googwoolf.ai/usage/naming-conventions)
  - [Prepare training data](https://docs.goodwoolf.ai/usage/prepare-data)
  - [Download model weights](https://docs.googwoolf.ai/usage/download-model)
  - [Train model](https://docs.googwoolf.ai/usage/train-model)
  - [Generate text](https://docs.googwoolf.ai/usage/generate-text)
  - [Create a new model](https://docs.googwoolf.ai/usage/create-model)
  - [Publish a new model](https://docs.googwoolf.ai/usage/publish-model)
  - [Publish a model update](https://docs.googwoolf.ai/usage/publish-update)

You are welcome to contribute to training of an existing model or to launch a brand-new one.

### Project Structure

The project consists of two repositories:

- [GoodWoolf Core](https://github.com/GoodWoolf-AI/GoodWoolf) (this repository) - the training framework
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

## Usage Examples

### Prepare training data

```
goodwoolf prepare <vendor>/<model> <dataset>
```

### Download model weights

```bash
goodwoolf pull <vendor>/<model>
```

### Train model

```bash
goodwoolf pretrain <vendor>/<model> <dataset>
```

### Generate text

```bash
goodwoolf run <vendor>/<model> --prompt "A long time ago..."
```

## License
- [GoodWoolf Core](https://github.com/GoodWoolf-AI/GoodWoolf): MIT
- [GoodWoolf Training Ledger](https://hf.co/datasets/GoodWoolf/Training-Ledger): OpenMDW