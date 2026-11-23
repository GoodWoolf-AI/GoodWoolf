"""Generic helpers for training and evaluation."""

import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from goodwoolf.utils.models import load_model
from goodwoolf.utils.system import tokenizer_path

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
        return torch.float32, nullcontext()
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


def load_model_from_file(model_name, path):
    return load_model(model_name, path)


def load_model_state(model_name, pretrained_path, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Resuming model from {checkpoint_path}", file=sys.stderr)
        model = load_model_from_file(model_name, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return model, checkpoint
    if os.path.exists(pretrained_path):
        print(f"Initializing model from {pretrained_path}", file=sys.stderr)
        return load_model_from_file(model_name, pretrained_path), None
    print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
    sys.exit(1)


def restore_progress(checkpoint, examples_per_update, total_examples, default_update=0):
    examples_consumed = checkpoint.get("examples_consumed", 0) if checkpoint else 0
    update_count = checkpoint.get("update_count", checkpoint.get("iter_num", default_update)) if checkpoint else default_update
    return min(examples_consumed, total_examples), update_count


ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


def extract_answer(completion):
    if not completion:
        return None
    match = ANSWER_RE.search(completion)
    return match.group(1).replace(",", "") if match else None


def evaluate(reference_answer, completion):
    prediction = extract_answer(completion)
    return int(prediction is not None and prediction == reference_answer)


def reward(reference_answer, completion):
    return float(evaluate(reference_answer, completion))


class ExampleUpdateTracker:
    def __init__(self, examples_per_update, total_examples, examples_consumed=0, update_count=0):
        if examples_per_update <= 0:
            raise ValueError("examples_per_update must be positive")
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")
        if examples_consumed < 0 or examples_consumed > total_examples:
            raise ValueError("examples_consumed must be between zero and total_examples")
        if update_count < 0:
            raise ValueError("update_count must be non-negative")
        self.examples_per_update = examples_per_update
        self.total_examples = total_examples
        self.examples_consumed = examples_consumed
        self.update_count = update_count

    @property
    def done(self):
        return self.examples_consumed >= self.total_examples

    @property
    def progress(self):
        return min(self.examples_consumed / self.total_examples, 1.0)

    @property
    def total_updates(self):
        return (self.total_examples + self.examples_per_update - 1) // self.examples_per_update

    def group_size(self):
        return min(self.examples_per_update, self.total_examples - self.examples_consumed)

    def consume(self, count):
        if count <= 0:
            raise ValueError("count must be positive")
        if self.examples_consumed + count > self.total_examples:
            raise ValueError("cannot consume more examples than total_examples")
        self.examples_consumed += count

    def finish_update(self):
        self.update_count += 1

    def state_dict(self):
        return {
            "examples_per_update": self.examples_per_update,
            "total_examples": self.total_examples,
            "examples_consumed": self.examples_consumed,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state):
        self.examples_per_update = state.get("examples_per_update", self.examples_per_update)
        self.total_examples = state.get("total_examples", self.total_examples)
        self.examples_consumed = state.get("examples_consumed", 0)
        self.update_count = state.get("update_count", 0)


def finish_gradient_update(scaler, optimizer, parameters, grad_clip, divisor):
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    params = list(parameters)
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
    for param in params:
        if param.grad is not None:
            param.grad.div_(divisor)
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def pack_rollout(sequences, masks, pad_token_id, device):
    max_length = max(len(sequence) for sequence in sequences)
    padded_sequences = [
        sequence + [pad_token_id] * (max_length - len(sequence))
        for sequence in sequences
    ]
    padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
    ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    mask_ids = torch.tensor(padded_masks, dtype=torch.bool, device=device)
    inputs = ids[:, :-1]
    targets = ids[:, 1:].clone()
    targets[~mask_ids[:, 1:]] = -1
    return inputs, targets


def _tokenizer_special_ids(tokenizer):
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    if special_ids:
        return special_ids
    get_special_tokens = getattr(tokenizer, "get_special_tokens", None)
    encode_special = getattr(tokenizer, "encode_special", None)
    if callable(get_special_tokens) and callable(encode_special):
        for token in get_special_tokens():
            special_ids.add(encode_special(token))
    return special_ids


def _decode_token(tokenizer, token_id):
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    if isinstance(text, list):
        text = "".join(str(part) for part in text)
    return text


def build_token_bytes(tokenizer, vocab_size):
    special_ids = _tokenizer_special_ids(tokenizer)
    values = []
    for token_id in range(int(vocab_size)):
        if token_id in special_ids:
            values.append(0)
            continue
        values.append(len(_decode_token(tokenizer, token_id).encode("utf-8")))
    return torch.tensor(values, dtype=torch.int32)


def save_token_bytes(model_name, token_bytes):
    path = Path(tokenizer_path(model_name))
    path.mkdir(parents=True, exist_ok=True)
    target = path / "token_bytes.pt"
    torch.save(token_bytes.to(dtype=torch.int32).cpu(), target)
    print(f"Saved token bytes to {target}", file=sys.stderr)
    return target


def load_token_bytes(model_name, device):
    path = Path(tokenizer_path(model_name)) / "token_bytes.pt"
    if not path.exists():
        raise FileNotFoundError(f"Token byte counts not found at {path}; run with --stage TOK first")
    token_bytes = torch.load(path, map_location=device)
    return token_bytes.to(device=device, dtype=torch.long)


def token_losses(model, inputs, targets, ctx=None):
    ctx = ctx or nullcontext()
    with ctx:
        output = model(input_ids=inputs, use_cache=False)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("Model did not return logits")
    if logits.shape[:2] != targets.shape:
        if logits.shape[0] == targets.shape[0] and logits.shape[1] == targets.shape[1] + 1:
            logits = logits[:, :-1, :]
        else:
            raise ValueError(f"Logits shape {tuple(logits.shape)} does not match targets {tuple(targets.shape)}")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    )
    return losses.view_as(targets)


def masked_mean(losses, targets):
    valid = targets >= 0
    return (losses * valid).sum() / valid.sum().clamp(min=1)


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, ctx=None):
    ctx = ctx or nullcontext()
    device = token_bytes.device
    total_nats = torch.zeros((), dtype=torch.float32, device=device)
    total_bytes = torch.zeros((), dtype=torch.int64, device=device)
    batch_iter = iter(batches)

    for _ in range(steps):
        try:
            inputs, targets = next(batch_iter)
        except StopIteration:
            break
        targets = targets.to(device)
        losses = token_losses(model, inputs, targets, ctx).to(device)
        valid = targets >= 0
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        num_bytes = torch.where(
            valid,
            token_bytes[safe_targets],
            torch.zeros_like(targets, dtype=token_bytes.dtype),
        )
        total_nats += (losses * (num_bytes > 0).to(losses.dtype)).sum()
        total_bytes += num_bytes.sum()

    total_bytes_value = total_bytes.item()
    if total_bytes_value == 0:
        return float("inf")
    return total_nats.item() / (math.log(2) * total_bytes_value)
