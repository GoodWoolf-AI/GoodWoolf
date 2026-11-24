#!/usr/bin/env python3
"""Ledger utilities for GoodWoolf."""

import json
import os
import uuid
from pathlib import Path

from goodwoolf.utils.system import (
    checkpoint_path,
    copy_dir,
    model_path,
    resolve_url,
    hash_files,
    download_file,
    rm_dir
)


LEDGER_ENTRY = {
    "id": None,
    "provider": None,
    "url": None,
    "weights": [],
    "inputs": [],
    "operator": None
}

OP_RND = "RND"  # Randomized
OP_NTP = "NTP"  # Next Token Prediction
OP_SFT = "SFT"  # Supervised Fine-Tuning
OP_RL = "RL"  # Reinforcement Learning
OP_AGG = "AGG"  # Aggregation

LEDGER_OPERATORS = [
    OP_RND,
    OP_NTP,
    OP_SFT,
    OP_RL,
    OP_AGG,
]

LEDGER_FILE_NAME = "training.ledger.json"
INDEX_FILE_NAME = "model.safetensors.index.json"


def get_hf_username():
    # Initialize the API client
    from huggingface_hub import HfApi

    api = HfApi()
    user_info = api.whoami()
    return user_info["name"]


def list_model_files(model_path):
    if not os.path.isdir(model_path):
        return None

    allowed_ext = {".safetensors"}
    allowed_files = {INDEX_FILE_NAME}

    model_files = []
    for d in Path(model_path).iterdir():
        if d.name in allowed_files or d.suffix in allowed_ext:
            model_files.append(d.name)
    return sorted(model_files)


def find_entry(entries, id=None, ops=None):
    # filter by id
    if id:
        filtered = []
        for entry in entries:
            if entry["id"] == id:
                filtered.append(entry)
        entries = filtered

    # filter by operator
    if ops:
        filtered = []
        for entry in entries:
            if entry["operator"] in ops:
                filtered.append(entry)
        entries = filtered

    # get the last match
    if len(entries) > 0:
        return entries[-1]

    return None


def create_entry(model_name: str, operator: str):
    """Add an entry to the training ledger."""
    if operator not in LEDGER_OPERATORS:
        raise ValueError(f"Invalid ledger operator {operator}.")

    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    # get HF username
    user_name = get_hf_username()

    # compute model hash
    model_dir = model_path(model_name)
    model_files = list_model_files(model_dir)
    model_hash = hash_files([f"{model_dir}/{f}" for f in model_files])

    entry = LEDGER_ENTRY.copy()
    entry["id"] = model_hash
    entry["provider"] = user_name
    entry["url"] = "<your_model_url>"
    entry["weights"] = model_files
    entry["operator"] = operator

    if ledger_file.exists():
        with open(ledger_file, "r") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            entries = [entries]
        if find_entry(entries, id=model_hash):
            raise Exception(f"Entry id {model_hash} already exists in {ledger_file}.")
        if operator == OP_RND and find_entry(entries, ops=[OP_RND]):
            raise Exception(f"Entry operator {OP_RND} already exists in {ledger_file}.")
        entries.append(entry)
    else:
        entries = [entry]

    with open(ledger_file, "w") as f:
        json.dump(entries, f, indent=2)

    return entry


def check_entry(model_name):
    """Check if the entry id matches the hash of the weights."""
    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    print("Ledger path:", ledger_file)

    # compute model hash
    model_dir = model_path(model_name)
    model_files = list_model_files(model_dir)
    model_hash = hash_files([f"{model_dir}/{f}" for f in model_files])

    # check if model hash is in the ledger
    with open(ledger_file, "r") as f:
        entries = json.load(f)
    entry = find_entry(entries, id=model_hash)
    if not entry:
        raise Exception(f"Model hash not found in the ledger.")

    # check url
    model_url = entry["url"]
    if not resolve_url(model_url):
        raise Exception(f"Invalid entry URL {model_url} in {ledger_file}.")

    # check weights against the model files
    if model_files != entry["weights"]:
        raise Exception(f"Model file names do not match ledger weight names.")

    # check operator
    op = entry["operator"]
    if op not in LEDGER_OPERATORS:
        raise Exception(f"Invalid entry operator {op} in {ledger_file}.")

    # check inputs
    inputs = entry["inputs"]
    if op == OP_RND and len(inputs) != 0:
        raise Exception(f"Operator {op} requires no inputs in {ledger_file}.")
    if op in (OP_NTP, OP_SFT, OP_RL) and len(inputs) != 1:
        raise Exception(f"Operator {op} requires a single input in {ledger_file}.")
    if op == OP_AGG and len(inputs) < 2:
        raise Exception(f"Operator {op} requires at least two inputs in {ledger_file}.")

    # check input references
    for id in inputs:
        entry = find_entry(entries, id)
        if entry is None:
            raise Exception(f"Input id {id} not found in {ledger_file}.")
        elif entry == model_hash:
            raise Exception(f"Input id {id} references the entry itself in {ledger_file}.")


def pull_model(model_name, model_id=None):
    """Pull model files defined by the model id."""
    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    with open(ledger_file, "r") as f:
        entries = json.load(f)

    entry = find_entry(entries, id=model_id, ops=[OP_RND, OP_AGG])
    if not entry:
        raise Exception(f"Model hash not found in the ledger.")

    model_url = entry["url"]
    model_files = entry["weights"]
    checkpoint_dir = checkpoint_path(model_name, entry["id"])

    print("Ledger path:", ledger_file)
    print("Entry id:", entry["id"])
    print("Model URL:", model_url)
    print("Checkpoint path:", checkpoint_dir)

    if Path(checkpoint_dir).exists():
        print("Model already downloaded.")
        return

    for file in model_files:
        download_file(f"{model_url}/{file}", f"{checkpoint_dir}/{file}", progress=True)


def model_checkout(model_name, model_id):
    """Check out a model checkpoint as the active model."""
    source_dir = checkpoint_path(model_name, model_id)
    if not Path(source_dir).is_dir():
        raise FileNotFoundError(f"Model checkpoint not pulled: {source_dir}. Run pull first.")
    target_dir = model_path(model_name)
    print("Checkout path:", target_dir)
    rm_dir(target_dir)
    copy_dir(source_dir, target_dir)
