#!/usr/bin/env python3
"""Manages models, configs and weights."""

import os
import sys
import torch
import yaml

from pathlib import Path
from huggingface_hub import HfApi
from collections import defaultdict

from goodwoolf.utils.system import copy_dir, models_path

HF_API = HfApi()
BASE_FILTERS = {"filter": "transformers", "pipeline_tag": "text-generation"}

CONFIG_BASE = Path(__file__).parents[1] / "config"
DATASET_BASE = Path(__file__).parents[1] / "data"
MODEL_BASE = Path(__file__).parents[1] / "models"


def model_name_from_path(path_name):
    # Resolve the full path and find its relative path to MODEL_BASE
    full_path = Path(path_name).resolve()
    relative_path = full_path.relative_to(MODEL_BASE.resolve())

    # Extract the first two parts (folders) of the remaining path
    vendor = relative_path.parts[0]
    name = relative_path.parts[1]

    return f"{vendor}/{name}"


def list_remote_vendors():
    vendors = defaultdict(int)
    for model in HF_API.list_models(**BASE_FILTERS, limit=500):
        author = model.modelId.split("/")[0] if "/" in model.modelId else "unknown"
        vendors[author] += 1
    print(f"Vendors:")
    for author in sorted(vendors, key=lambda a: -vendors[a]):
        print(f"{author:40s} {vendors[author]:>6d} models")


def list_remote_models(vendor):
    kwargs = {**BASE_FILTERS}
    if vendor:
        kwargs["search"] = vendor
    print(f"Models:")
    for model in HF_API.list_models(**kwargs):
        if vendor and not model.modelId.split("/")[0].lower() == vendor.lower():
            continue
        print(model.modelId)


def list_vendors():
    """List supported datasets (subdirectories in DATASET_BASE)."""
    print(f"Vendors:")
    path = MODEL_BASE
    if not path.exists():
        return
    excluded = {"__pycache__"}
    for d in path.iterdir():
        if d.is_dir() and d.name not in excluded:
            print(d.name)


def list_models(vendor):
    """List supported vendor models (subdirectories in MODEL_BASE/vendor)."""
    print("Models:")
    path = Path(MODEL_BASE / vendor)
    if not path.exists():
        return
    excluded = {"__pycache__"}
    for d in path.iterdir():
        if d.is_dir() and d.name not in excluded:
            print(f"{vendor}/{d.name}")


def list_datasets():
    """List supported datasets (subdirectories in DATASET_BASE)."""
    print(f"Datasets:")
    if not DATASET_BASE.exists():
        return
    excluded = {"__pycache__"}
    for d in DATASET_BASE.iterdir():
        if d.is_dir() and d.name not in excluded:
            print(d.name)


def list_weights(path):
    from safetensors import safe_open

    total = 0
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            dims = f.get_slice(name).get_shape()
            nelems = 1
            for d in dims:
                nelems *= d
            print(f"{name:60s} {str(dims):20s} {nelems:>12,}")
            total += nelems
    print(f"{'total':60s} {'':20s} {total:>12,}")


def load_config(model_name):
    from transformers import AutoConfig

    print(f"Loading config for {model_name}", file=sys.stderr)
    yaml_path = CONFIG_BASE / model_name / "model.yaml"
    with open(yaml_path) as f:
        kwargs = yaml.safe_load(f)
    model_type = kwargs.pop("model_type", "llama")
    model_config = AutoConfig.for_model(model_type, **kwargs)

    tokenizer = load_tokenizer(model_name)
    return {"model": model_config, "tokenizer": tokenizer}


def create_model(model_name, config, template, weights):
    if config:
        save_config(model_name)
    if template:
        print("Creating pipeline files:", template, "->", model_name)
        copy_dir(MODEL_BASE / template, MODEL_BASE / model_name)
    if weights:
        save_model(model_name, models_path(model_name))


def save_config(model_name):
    from transformers import AutoConfig, AutoTokenizer

    print(f"Fetching config for {model_name}", file=sys.stderr)
    model_config = AutoConfig.from_pretrained(model_name)

    path = CONFIG_BASE / model_name
    path.mkdir(parents=True, exist_ok=True)
    yaml_path = path / "model.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(model_config.to_dict(), f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {yaml_path}", file=sys.stderr)

    print(f"Fetching tokenizer for {model_name}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(path))
    print(f"Saved tokenizer to {path}", file=sys.stderr)


def load_tokenizer(model_name):
    tokenizer_path = MODEL_BASE / model_name / "tokenizer.py"
    if tokenizer_path.exists():
        from goodwoolf.utils.system import import_file

        mod = import_file(tokenizer_path, f"tokenizer_{model_name}")
        print(f"Creating tokenizer from {tokenizer_path}", file=sys.stderr)
        return mod.Tokenizer(model_name)

    from transformers import AutoTokenizer
    config_path = CONFIG_BASE / model_name
    if config_path.exists() and (config_path / "tokenizer.json").exists():
        print(f"Loading tokenizer from {config_path}", file=sys.stderr)
        return AutoTokenizer.from_pretrained(str(config_path))
    return None


def new_model(model_name):
    model_path = MODEL_BASE / model_name / "model.py"
    if model_path.exists():
        from goodwoolf.utils.system import import_file

        mod = import_file(model_path, f"model_{model_name}")
        configs = load_config(model_name)
        print(f"Creating model from {model_path}", file=sys.stderr)
        return mod.Model(configs["model"])

    from transformers import AutoModelForCausalLM

    config_path = CONFIG_BASE / model_name
    if config_path.exists():
        configs = load_config(model_name)
        print(f"Creating model from config", file=sys.stderr)
        return AutoModelForCausalLM.from_config(configs["model"])
    else:
        print(f"Loading pretrained model {model_name}", file=sys.stderr)
        return AutoModelForCausalLM.from_pretrained(model_name)


def save_safetensors(state_dict, save_directory, max_shard_size="2GB"):
    from huggingface_hub import save_torch_state_dict

    Path(save_directory).mkdir(parents=True, exist_ok=True)

    save_torch_state_dict(
        state_dict=state_dict,
        save_directory=save_directory,
        max_shard_size=max_shard_size,
        safe_serialization=True  # Enforces .safetensors format
    )

def load_safetensors(load_directory):
    import os
    import json
    from safetensors.torch import load_file

    index_path = os.path.join(load_directory, "model.safetensors.index.json")
    single_file_path = os.path.join(load_directory, "model.safetensors")

    # If no index json exists, load a single-file save
    if not os.path.exists(index_path):
        if os.path.exists(single_file_path):
            return load_file(single_file_path)
        else:
            raise FileNotFoundError(
                f"Could not find a valid safetensors model in {load_directory} "
                f"(neither an index json nor a single 'model.safetensors' exists)."
            )

    # Load sharded files using the index map
    with open(index_path, "r") as f:
        index_data = json.load(f)

    unique_shards = set(index_data["weight_map"].values())

    full_state_dict = {}
    for shard in unique_shards:
        shard_path = os.path.join(load_directory, shard)
        full_state_dict.update(load_file(shard_path))

    return full_state_dict


def save_model(model_name, save_path, load_path=None):
    """Save model optionally converting weights to a different format."""
    if load_path:
        model = load_model(model_name, load_path)
    else:
        model = new_model(model_name)

    print(f"Saving weights to {save_path}", file=sys.stderr)
    state_dict = {k: v.contiguous().clone() for k, v in model.state_dict().items()}
    save_safetensors(state_dict, save_path)


def load_model(model_name, load_path):
    """Load model with weights in different formats."""
    from safetensors.torch import load_file
    model = new_model(model_name)

    print(f"Loading weights from {load_path}", file=sys.stderr)
    if os.path.isdir(load_path):
        state_dict = load_safetensors(load_path)
    elif load_path.endswith(".safetensors"):
        state_dict = load_file(load_path)
    else:
        ckpt = torch.load(load_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        for k in list(state_dict.keys()):
            if k.startswith("_orig_mod."):
                state_dict[k[10:]] = state_dict.pop(k)

    model.load_state_dict(state_dict, strict=False)
    return model


@torch.inference_mode()
def generate(model_name, load_path, prompt, max_tokens):
    """Generate text from a prompt using the model."""
    configs = load_config(model_name)
    model = load_model(model_name, load_path)
    model.eval()

    tokenizer = configs["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt")

    # Ensure input prompt has at least one token
    if inputs["input_ids"].shape[1] == 0:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError(
                "Cannot generate from an empty prompt: "
                "the tokenizer has no BOS token."
            )
        inputs["input_ids"] = torch.tensor(
            [[bos_token_id]],
            dtype=torch.long,
            device=inputs["input_ids"].device,
        )
        inputs["attention_mask"] = torch.ones_like(
            inputs["input_ids"]
        )

    # Move variables and attributes together
    # inputs {
    #     "input_ids": tensor[...],
    #     "attention_mask": tensor[...]
    # }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


@torch.no_grad()
def estimate_loss(model, ctx, device, data_dir, block_size, batch_size, eval_iters, device_type):
    """Estimate loss for train and test splits."""
    import numpy as np
    import torch

    out = {}
    model.eval()
    for split in ["train", "test"]:
        data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            with ctx:
                loss = model(input_ids=x, labels=y).loss
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def add_weights(model_name: str, base_model_path: str, new_model_path: str):
    """Add weight deltas from new model to base model."""
    import torch

    from goodwoolf.utils.models import load_model

    print(f"Loading base model from {base_model_path}", file=sys.stderr)
    base_model = load_model(model_name, base_model_path)

    print(f"Loading new model from {new_model_path}", file=sys.stderr)
    new_model = load_model(model_name, new_model_path)

    base_sd = base_model.state_dict()
    new_sd = new_model.state_dict()

    for k in new_sd:
        dw_k = new_sd[k] - base_sd[k]
        base_sd[k] = base_sd[k] + dw_k

    base_model.load_state_dict(base_sd)
    return base_model


def update(model_name, data_name, base_model, new_model, batch_size, eval_iters, dtype):
    """Update model with delta weights from new model."""
    import os
    import time
    from contextlib import nullcontext

    import numpy as np
    import torch

    from goodwoolf.utils.models import load_config

    data_dir = data_path(model_name, data_name)
    configs = load_config(model_name)
    model_config = configs["model"]

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    device_type = "cuda" if "cuda" in device and has_cuda else "cpu"
    compute_ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=compute_ptdtype)

    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.manual_seed(1337)
    block_size = model_config.max_position_embeddings

    base_model = load_model(model_name, base_model)
    base_model.to(device)

    print(f"data:    {data_dir}", file=sys.stderr)
    print(f"splits:  train, test", file=sys.stderr)
    print(f"iters:   {eval_iters} per split", file=sys.stderr)
    print(f"batch:   {batch_size} x {block_size} tokens", file=sys.stderr)
    print(file=sys.stderr)

    print("--- Base Model ---", file=sys.stderr)
    t0 = time.time()
    base_losses = estimate_loss(base_model, ctx, device, data_dir, block_size, batch_size, eval_iters, device_type)
    dt = time.time() - t0
    print(f"train loss: {base_losses['train']:.4f}", file=sys.stderr)
    print(f"test loss:  {base_losses['test']:.4f}", file=sys.stderr)
    print(f"time:       {dt:.2f}s", file=sys.stderr)

    print("--- New Model ---", file=sys.stderr)
    new_model = load_model(model_name, new_model)
    new_model.to(device)
    t0 = time.time()
    new_losses = estimate_loss(new_model, ctx, device, data_dir, block_size, batch_size, eval_iters, device_type)
    dt = time.time() - t0
    print(f"train loss: {new_losses['train']:.4f}", file=sys.stderr)
    print(f"test loss:  {new_losses['test']:.4f}", file=sys.stderr)
    print(f"time:       {dt:.2f}s", file=sys.stderr)

    assert new_losses["train"] < base_losses["train"], "New model train loss not lower"
    assert new_losses["test"] < base_losses["test"], "New model test loss not lower"

    print("--- Updating Weights ---", file=sys.stderr)
    updated_model = add_weights(model_name, base_model, new_model)

    output_path = base_model
    from safetensors.torch import save_file
    save_file(updated_model.state_dict(), output_path)
    print(f"Updated model saved to {output_path}", file=sys.stderr)
