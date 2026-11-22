"""Generic helpers for training and evaluation."""

import math
import re
from contextlib import nullcontext
from pathlib import Path

import torch

from goodwoolf.utils.system import tokenizer_path

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


def load_token_bytes(model_name, device):
    path = Path(tokenizer_path(model_name)) / "token_bytes.pt"
    if not path.exists():
        raise FileNotFoundError(f"Token byte counts not found at {path}; run tokenizer training first")
    token_bytes = torch.load(path, map_location=device)
    return token_bytes.to(device=device, dtype=torch.long)


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, ctx=None):
    ctx = ctx or nullcontext()
    device = token_bytes.device
    total_nats = torch.zeros((), dtype=torch.float32, device=device)
    total_bytes = torch.zeros((), dtype=torch.int64, device=device)
    batch_iter = iter(batches)

    for _ in range(steps):
        inputs, targets = next(batch_iter)
        with ctx:
            output = model(input_ids=inputs, labels=targets, loss_reduction="none")
        losses = output.loss.reshape(-1).to(device)
        targets = targets.reshape(-1).to(device)
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
