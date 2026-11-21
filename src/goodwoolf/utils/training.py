"""Generic helpers for RL training."""

import re

import torch

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
