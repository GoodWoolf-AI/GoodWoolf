#!/usr/bin/env python3
"""Ledger utilities for GoodWoolf."""

import json
import os
import uuid
from pathlib import Path

from goodwoolf.utils.system import (
    models_path,
    resolve_url,
    hash_files,
    download_file
)
from goodwoolf.utils.models import models_path


LEDGER_ENTRY = {
    "id": None,
    "provider": None,
    "url": None,
    "weights": [],
    "inputs": [],
    "operator": None
}

OP_RANDOMIZED = "randomized"
OP_PRETRAINED = "pretrained"
OP_FINETUNED = "finetuned"
OP_AGGREGATED = "aggregated"

LEDGER_OPERATORS = [
    OP_RANDOMIZED,
    OP_PRETRAINED,
    OP_FINETUNED,
    OP_AGGREGATED,
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
    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    # get HF username
    user_name = get_hf_username()

    # compute model hash
    model_path = models_path(model_name)
    model_files = list_model_files(model_path)
    model_hash = hash_files([f"{model_path}/{f}" for f in model_files])

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
        if operator == OP_RANDOMIZED and find_entry(entries, ops=[OP_RANDOMIZED]):
            raise Exception(f"Entry operator {OP_RANDOMIZED} already exists in {ledger_file}.")
        entries.append(entry)
    else:
        entries = [entry]

    with open(ledger_file, "w") as f:
        json.dump(entries, f, indent=2)

    return entry


def check_entry(model_name):
    """Check if the entry id matched hash of the wrights."""
    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    print("Ledger path:", ledger_file)

    # compute model hash
    model_path = models_path(model_name)
    model_files = list_model_files(model_path)
    model_hash = hash_files([f"{model_path}/{f}" for f in model_files])

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

    # check weights agains the model files
    if model_files != entry["weights"]:
        raise Exception(f"Model file names do not match ledger weight names.")

    # check operator
    op = entry["operator"]
    if op not in LEDGER_OPERATORS:
        raise Exception(f"Invalid entry operator {op} in {ledger_file}.")

    # check inputs
    inputs = entry["inputs"]
    if op == "randomized" and len(inputs) != 0:
        raise Exception(f"Operator {op} requires no inputs in {ledger_file}.")
    if op == "pretrained" and len(inputs) != 1:
        raise Exception(f"Operator {op} requires a single input in {ledger_file}.")
    if op == "finetuned" and len(inputs) != 1:
        raise Exception(f"Operator {op} requires a single input in {ledger_file}.")
    if op == "aggregated" and len(inputs) > 1:
        raise Exception(f"Operator {op} requires at least two inputs in {ledger_file}.")

    # check input references
    for id in inputs:
        entry = find_entry(entries, id)
        if entry is None:
            raise Exception(f"Input id {id} not found in {ledger_file}.")
        elif entry == model_hash:
            raise Exception(f"Input id {id} references the entry itself in {ledger_file}.")


def pull_model(model_name, id=None):
    """Pull model files defined by the entry id."""
    ledger_base = os.environ.get("GOODWOOLF_LEDGER")
    if not ledger_base:
        raise ValueError("Environment variable GOODWOOLF_LEDGER must be set")

    ledger_dir = Path(ledger_base) / model_name
    ledger_file = ledger_dir / LEDGER_FILE_NAME

    with open(ledger_file, "r") as f:
        entries = json.load(f)

    entry = find_entry(entries, id=id, ops=[OP_RANDOMIZED, OP_AGGREGATED])
    if not entry:
        raise Exception(f"Model hash not found in the ledger.")

    model_url = entry["url"]
    model_files = entry["weights"]
    model_path = models_path(model_name)

    print("Ledger path:", ledger_file)
    print("Entry id:", entry["id"])
    print("Model URL:", model_url)

    for file in model_files:
        download_file(f"{model_url}/{file}", f"{model_path}/{file}", progress=True)
