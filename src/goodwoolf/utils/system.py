#!/usr/bin/env python3
"""System utilities for GoodWoolf."""

import os
import importlib.util
import shutil
import sys
import socket
import hashlib
import requests

from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm

# Define a buffer size to read/download files in chunks (e.g., 64KB)
# This keeps memory usage low even for massive files
BUFFER_SIZE = 1024 * 1024  # 1 MB


def data_path(model_name, data_name):
    base = os.environ.get("GOODWOOLF_DATA", str(Path.home()) + "/.goodwoolf/data")
    if model_name is None or data_name is None:
        return base
    return base + "/" + data_name + "/" + model_name


def models_path(model_name=None):
    base = os.environ.get("GOODWOOLF_MODELS", str(Path.home()) + "/.goodwoolf/models")
    if model_name is None:
        return base
    return base + "/" + model_name


def copy_dir(src_dir, dst_dir):
    """Copy a folder from src_dir to dst_dir."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)
    for item in src_path.iterdir():
        dest = dst_path / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def import_file(file_path, package_name):
    """Import python package from path."""
    spec = importlib.util.spec_from_file_location(package_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def import_package(package_path, package_name):
    """Import python package from path"""
    path = Path(package_path) / "__init__.py"
    return import_file(path, package_name)


def resolve_url(url: str) -> bool:
    try:
        # Extract the hostname (e.g., "google.com") from the URL
        hostname = urlparse(url).netloc
        if not hostname:
            return False
            
        # Strip port number if present (e.g., "localhost:8080" -> "localhost")
        hostname = hostname.split(':')[0]
        
        # Attempt to look up the IP address
        socket.gethostbyname(hostname)
        return True
    except (socket.gaierror, ValueError):
        # socket.gaierror is raised if the domain cannot be resolved
        return False


def hash_files(file_list, verbose=True):
    """
    Calculates the SHA-1 checksum of multiple files combined,
    equivalent to: cat fileA fileB fileC | sha1sum
    """
    sha1 = hashlib.sha1()

    for file_path in file_list:
        if verbose:
            print("Hashing", file_path)
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(BUFFER_SIZE)
                if not data:
                    break
                sha1.update(data)

    return sha1.hexdigest()


def download_file(file_url, save_path, progress=None, create_path=True):
    """
    Download a file from a URL to a local path using streaming.
    """
    if create_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with requests.get(file_url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            bar = tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                disable=not progress,
                desc="Downloading",
            )

            try:
                for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            finally:
                bar.close()