"""Generic helpers for training and evaluation."""

import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

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


class BatchPacker:
    def __init__(
        self,
        source,
        bos_token_id,
        batch_size,
        row_capacity,
        buffer_size,
        device,
        device_type,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if row_capacity <= 0:
            raise ValueError("row_capacity must be positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.source = source
        self.bos_token_id = bos_token_id
        self.batch_size = batch_size
        self.row_capacity = row_capacity
        self.buffer_size = buffer_size
        self.device = device
        self.device_type = device_type
        self.buffer = []

    @property
    def done(self):
        return bool(getattr(self.source, "done", False)) and not self.buffer

    def _ensure_buffer(self):
        while len(self.buffer) < self.buffer_size and not getattr(self.source, "done", False):
            sample = self.source.next_sample()
            if sample is None:
                break
            ids, mask = sample
            if ids:
                self.buffer.append((ids, mask))

    def next_batch(self, max_examples=None):
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive")
        rows, mask_rows, row_lengths = [], [], []
        items_used = 0

        for _ in range(self.batch_size):
            row, mask_row, content_len = [], [], None
            pos = 0
            while pos < self.row_capacity:
                if max_examples is not None and items_used >= max_examples:
                    content_len = pos
                    break
                self._ensure_buffer()
                if not self.buffer:
                    content_len = pos
                    break
                remaining = self.row_capacity - pos
                best_idx = -1
                best_len = 0
                for idx, (ids, _) in enumerate(self.buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx = idx
                        best_len = len(ids)
                if best_idx >= 0:
                    ids, mask = self.buffer.pop(best_idx)
                    take = min(len(ids), remaining)
                    row.extend(ids[:take])
                    mask_row.extend(mask[:take])
                    pos += take
                    items_used += 1
                    continue
                shortest_idx = min(range(len(self.buffer)), key=lambda i: len(self.buffer[i][0]))
                ids, mask = self.buffer.pop(shortest_idx)
                take = min(len(ids), remaining)
                row.extend(ids[:take])
                mask_row.extend(mask[:take])
                pos += take
                items_used += 1
                content_len = self.row_capacity
                break
            if pos < self.row_capacity:
                pad = self.row_capacity - pos
                row.extend([self.bos_token_id] * pad)
                mask_row.extend([0] * pad)
                if content_len is None:
                    content_len = pos
            row_lengths.append(self.row_capacity if content_len is None or content_len < 0 else content_len)
            rows.append(row[:self.row_capacity])
            mask_rows.append(mask_row[:self.row_capacity])

        if items_used == 0:
            return None
        use_cuda = self.device_type == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(self.device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(self.device, non_blocking=use_cuda).clone()
        mask_tensor = torch.tensor(mask_rows, dtype=torch.bool, device=self.device)
        targets[~mask_tensor[:, 1:]] = -1
        for row_idx, content_len in enumerate(row_lengths):
            if content_len < self.row_capacity:
                targets[row_idx, max(0, content_len - 1):] = -1
        return inputs, targets, items_used, self.done

    def state_dict(self):
        return {
            "buffer": [
                (list(ids), list(mask))
                for ids, mask in self.buffer
            ],
        }

    def load_state_dict(self, state):
        self.buffer = [
            (list(ids), list(mask))
            for ids, mask in state.get("buffer", [])
        ]


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
