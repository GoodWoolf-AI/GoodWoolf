#!/usr/bin/env python3
"""Podman machine and container helpers."""

from pathlib import Path
import subprocess

DEFAULT_MACHINE_NAME = "goodwoolf-machine"
DEFAULT_CONNECTION_NAME = "goodwoolf-machine"
DEFAULT_CONTAINER_NAME = "goodwoolf"
DEFAULT_IMAGE = "docker.io/library/ubuntu"
DEFAULT_CONTAINER_DIR = "/workspace"


def run_podman(*args, check=True, capture_output=False):
    command = ["podman", *[str(arg) for arg in args]]
    print(" ".join(command))
    return subprocess.run(command, check=check, capture_output=capture_output, text=True)


def machine_create(cpus=None, mem=None, disk=None, size=None):
    args = ["machine", "init"]
    if cpus is not None:
        args.extend(["--cpus", cpus])
    if mem is not None:
        args.extend(["--memory", mem * 1024])
    if size is not None:
        args.extend(["--swap", size * 1024])
    if disk is not None:
        args.extend(["--disk-size", disk])
    args.append(DEFAULT_MACHINE_NAME)
    run_podman(*args)


def machine_start():
    run_podman("machine", "start", DEFAULT_MACHINE_NAME)


def machine_stop():
    run_podman("machine", "stop", DEFAULT_MACHINE_NAME)


def machine_shell():
    run_podman("machine", "ssh", DEFAULT_MACHINE_NAME)


def machine_remove():
    run_podman("machine", "rm", "-f", DEFAULT_MACHINE_NAME)


def container_create(cpus=None, mem=None, disk=None, size=None):
    args = ["--connection", DEFAULT_CONNECTION_NAME, "create"]
    if cpus is not None:
        args.extend(["--cpus", cpus])
    args.extend(
        [
            "--name",
            DEFAULT_CONTAINER_NAME,
            "--volume",
            f"{Path.cwd()}:{DEFAULT_CONTAINER_DIR}",
            "--workdir",
            DEFAULT_CONTAINER_DIR,
        ]
    )
    if mem is not None:
        args.extend(["--memory", f"{mem * 1024}m"])
    if size is not None:
        args.extend(["--memory-swap", f"{((mem or 0) + size) * 1024}m"])
    if disk is not None:
        args.extend(["--storage-opt", f"size={disk}G"])
    args.extend(["--tty", "--interactive", DEFAULT_IMAGE])
    run_podman(*args)


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
