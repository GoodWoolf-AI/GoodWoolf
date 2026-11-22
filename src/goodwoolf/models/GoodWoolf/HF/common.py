import math
import os
import sys

import torch

from goodwoolf.utils.models import load_model


def resolve_device(train_config):
    has_cuda = torch.cuda.is_available()
    device = train_config.device or ("cuda" if has_cuda else "cpu")
    device_type = "cuda" if "cuda" in device and has_cuda else "cpu"
    return device, device_type


def make_context(device_type, dtype_name):
    compute_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    if device_type == "cpu":
        return torch.float32, __import__("contextlib").nullcontext()
    return compute_dtype, torch.amp.autocast(device_type=device_type, dtype=compute_dtype)


def configure_optimizers(model, train_config, device_type):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": train_config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        fused=device_type == "cuda",
    )


def get_lr(progress, train_config):
    tc = train_config
    if tc.warmup_ratio > 0 and progress < tc.warmup_ratio:
        return tc.learning_rate * progress / tc.warmup_ratio
    if progress <= 1.0 - tc.warmdown_ratio:
        return tc.learning_rate
    denominator = max(tc.warmdown_ratio, 1e-8)
    decay_ratio = min((progress - (1.0 - tc.warmdown_ratio)) / denominator, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_model_state(model_name, load_model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Resuming model from {checkpoint_path}", file=sys.stderr)
        model = load_model_from_file(model_name, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return model, checkpoint
    if os.path.exists(load_model):
        print(f"Initializing model from {load_model}", file=sys.stderr)
        return load_model_from_file(model_name, load_model), None
    print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
    sys.exit(1)


def load_model_from_file(model_name, path):
    return load_model(model_name, path)


def restore_progress(checkpoint, examples_per_update, total_examples, default_update=0):
    examples_consumed = checkpoint.get("examples_consumed", 0) if checkpoint else 0
    update_count = checkpoint.get("update_count", checkpoint.get("iter_num", default_update)) if checkpoint else default_update
    return min(examples_consumed, total_examples), update_count


def get_pad_token_id(tokenizer):
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return token_id
    return 0


def generate_samples(
    model,
    tokenizer,
    prompt_ids,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    seed,
    device,
):
    generator = getattr(model, "generate", None)
    if generator is None:
        raise RuntimeError("Model does not provide generate()")
    torch.manual_seed(seed)
    try:
        result = generator(
            tokenizer,
            prompt_ids,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, [[1] * len(sequence) for sequence in result]
    except TypeError:
        pass

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-6),
        "num_return_sequences": num_samples,
        "pad_token_id": get_pad_token_id(tokenizer),
    }
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if top_k and top_k > 0:
        kwargs["top_k"] = top_k
    output = generator(input_ids=input_ids, **kwargs)
    sequences = output.sequences if hasattr(output, "sequences") else output
    rows = sequences[:, len(prompt_ids):].tolist()
    full_sequences = [list(prompt_ids) + row for row in rows]
    masks = [[0] * len(prompt_ids) + [1] * len(row) for row in rows]
    return full_sequences, masks
