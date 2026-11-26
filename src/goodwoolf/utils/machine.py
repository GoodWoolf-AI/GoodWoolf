#!/usr/bin/env python3
"""Podman machine and container helpers."""

from pathlib import Path
import subprocess

DEFAULT_MACHINE_NAME = "goodwoolf-machine"
DEFAULT_CONNECTION_NAME = "goodwoolf-machine"
DEFAULT_CONTAINER_NAME = "goodwoolf"
DEFAULT_IMAGE = "docker.io/library/ubuntu"
DEFAULT_CONTAINER_DIR = "/workspace"
DEFAULT_CPUS = 2
DEFAULT_MEMORY = 4
DEFAULT_DISK_SIZE = 100
DEFAULT_SWAP_SIZE = 90


def run_podman(*args, check=True, capture_output=False):
    command = ["podman", *[str(arg) for arg in args]]
    print(" ".join(command))
    return subprocess.run(command, check=check, capture_output=capture_output, text=True)


def machine_create(
    cpus=DEFAULT_CPUS,
    disk_size=DEFAULT_DISK_SIZE,
    memory=DEFAULT_MEMORY,
    swap_size=DEFAULT_SWAP_SIZE,
):
    if cpus is None or disk_size is None or memory is None or swap_size is None:
        raise RuntimeError("--cpus, --disk-size, --memory, and --swap-size are required")
    run_podman(
        "machine",
        "init",
        "--cpus",
        cpus,
        "--memory",
        memory * 1024,
        "--swap",
        swap_size * 1024,
        "--disk-size",
        disk_size,
        DEFAULT_MACHINE_NAME,
    )


def machine_start():
    run_podman("machine", "start", DEFAULT_MACHINE_NAME)


def machine_stop():
    run_podman("machine", "stop", DEFAULT_MACHINE_NAME)


def machine_shell():
    run_podman("machine", "ssh", DEFAULT_MACHINE_NAME)


def machine_remove():
    run_podman("machine", "rm", "-f", DEFAULT_MACHINE_NAME)


def container_create(
    cpus=DEFAULT_CPUS,
    disk_size=DEFAULT_DISK_SIZE,
    memory=DEFAULT_MEMORY,
    swap_size=DEFAULT_SWAP_SIZE,
):
    if cpus is None or disk_size is None or memory is None or swap_size is None:
        raise RuntimeError("--cpus, --disk-size, --memory, and --swap-size are required")
    run_podman(
        "--connection",
        DEFAULT_CONNECTION_NAME,
        "create",
        "--cpus",
        cpus,
        "--name",
        DEFAULT_CONTAINER_NAME,
        "--volume",
        f"{Path.cwd()}:{DEFAULT_CONTAINER_DIR}",
        "--workdir",
        DEFAULT_CONTAINER_DIR,
        "--memory",
        f"{memory * 1024}m",
        "--memory-swap",
        f"{(memory + swap_size) * 1024}m",
        "--storage-opt",
        f"size={disk_size}G",
        "--tty",
        "--interactive",
        DEFAULT_IMAGE,
    )


def container_start():
    run_podman("--connection", DEFAULT_CONNECTION_NAME, "start", DEFAULT_CONTAINER_NAME)


def container_stop():
    run_podman("--connection", DEFAULT_CONNECTION_NAME, "stop", DEFAULT_CONTAINER_NAME)


def container_shell():
    result = run_podman(
        "--connection",
        DEFAULT_CONNECTION_NAME,
        "inspect",
        "--format",
        "{{.State.Running}}",
        DEFAULT_CONTAINER_NAME,
        check=False,
        capture_output=True,
    )
    if result.stdout.strip() != "true":
        run_podman("--connection", DEFAULT_CONNECTION_NAME, "start", DEFAULT_CONTAINER_NAME)
    run_podman("--connection", DEFAULT_CONNECTION_NAME, "exec", "-it", DEFAULT_CONTAINER_NAME, "bash")


def container_remove():
    run_podman("--connection", DEFAULT_CONNECTION_NAME, "rm", "-f", DEFAULT_CONTAINER_NAME)
