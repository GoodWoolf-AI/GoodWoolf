#!/usr/bin/env python3
"""Podman machine and container helpers."""

import json
import os
import select
import shutil
import subprocess
import sys
import termios
import threading
import tty
from pathlib import Path

from podman import PodmanClient
from podman.errors import ImageNotFound

DEFAULT_MACHINE_NAME = "goodwoolf-machine"
DEFAULT_CONTAINER_NAME = "goodwoolf"
DEFAULT_IMAGE = "docker.io/library/ubuntu:24.04"
DEFAULT_CONTAINER_DIR = "/workspace"
DEFAULT_CPUS = 2
HELPER_CONFIG_NAME = ".podman-helpers.conf"


def _helper_config():
    return Path.cwd() / HELPER_CONFIG_NAME


def _find_executable(name):
    path = shutil.which(name)
    if path is None and name == "virtiofsd":
        candidate = Path("/usr/libexec/virtiofsd")
        if candidate.exists() and os.access(candidate, os.X_OK):
            path = str(candidate)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH")
    return path


def _machine_env():
    env = os.environ.copy()
    config = _helper_config()
    if config.exists():
        env["CONTAINERS_CONF_OVERRIDE"] = str(config)
    return env


def _run_podman(args, *, env=None, check=True, stdin=None):
    command = ["podman", *[str(arg) for arg in args]]
    return subprocess.run(
        command,
        check=check,
        env=env or _machine_env(),
        input=stdin,
        text=stdin is not None,
    )


def _machine_state(machine_name, env=None):
    result = subprocess.run(
        [
            "podman",
            "machine",
            "inspect",
            "--format",
            "{{.State}}",
            machine_name,
        ],
        check=True,
        env=env or _machine_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write_helper_config(gvproxy_path):
    config = _helper_config()
    helper_dir = str(Path(gvproxy_path).parent)
    config.write_text(f'[engine]\nhelper_binaries_dir = ["{helper_dir}"]\n')
    print(f"Wrote {config} (helper_binaries_dir={helper_dir})")


def check_machine():
    paths = {
        name: _find_executable(name)
        for name in ("podman", "gvproxy", "virtiofsd")
    }
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def setup_machine(
    machine_name=DEFAULT_MACHINE_NAME,
    cpus=DEFAULT_CPUS,
    disk_size=None,
    ram_size=None,
    swap_size=None,
):
    if disk_size is None or ram_size is None or swap_size is None:
        raise RuntimeError("--disk-size, --ram-size, and --swap-size are required")

    paths = check_machine()
    _write_helper_config(paths["gvproxy"])
    env = _machine_env()

    existing = subprocess.run(
        ["podman", "machine", "inspect", machine_name],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if existing.returncode == 0:
        raise RuntimeError(
            f"Podman machine '{machine_name}' already exists. "
            f"Remove it first with: gw machine clear"
        )

    print(f"Initializing Podman machine '{machine_name}'")
    _run_podman(
        [
            "machine",
            "init",
            "--cpus",
            cpus,
            "--memory",
            ram_size,
            "--swap",
            0,
            "--disk-size",
            disk_size,
            machine_name,
        ],
        env=env,
    )

    state = _machine_state(machine_name, env)
    if state == "running":
        print(f"Machine '{machine_name}' already running.")
    else:
        print(f"Starting '{machine_name}'...")
        _run_podman(["machine", "start", machine_name], env=env)

    swap_script = f"""set -euo pipefail
if swapon --show=NAME --noheadings | grep -q /var/swapfile; then
    echo "Swap already active on /var/swapfile"
    swapon --show
    exit 0
fi
FS_TYPE=$(df -T /var | awk 'NR==2 {{print $2}}')
echo "Detected /var filesystem: $FS_TYPE"
if [ ! -f /var/swapfile ]; then
    if [ "$FS_TYPE" = "btrfs" ]; then
        btrfs filesystem mkswapfile --size {swap_size}g /var/swapfile
    else
        fallocate -l {swap_size}G /var/swapfile
        chmod 600 /var/swapfile
        mkswap /var/swapfile
    fi
fi
grep -q '/var/swapfile' /etc/fstab || echo '/var/swapfile none swap sw 0 0' >> /etc/fstab
swapon /var/swapfile
swapon --show
"""
    print(f"Creating and enabling {swap_size}G disk swap inside '{machine_name}'...")
    _run_podman(
        ["machine", "ssh", machine_name, "sudo", "bash", "-s"],
        env=env,
        stdin=swap_script,
    )
    print(f"Done! Machine '{machine_name}' is ready.")


def open_machine(machine_name=DEFAULT_MACHINE_NAME):
    env = _machine_env()
    state = _machine_state(machine_name, env)
    if state != "running":
        _run_podman(["machine", "start", machine_name], env=env)
    _run_podman(["machine", "ssh", machine_name], env=env)


def clear_machine(machine_name=DEFAULT_MACHINE_NAME):
    _run_podman(["machine", "rm", "-f", machine_name], check=False)
    config = _helper_config()
    if config.exists():
        config.unlink()
        print(f"Removed {config}")


def _container_client(machine_name):
    config = _helper_config()
    if config.exists():
        os.environ.setdefault("CONTAINERS_CONF_OVERRIDE", str(config))
    try:
        return PodmanClient(connection=machine_name)
    except KeyError as exc:
        raise RuntimeError(f"Podman connection not found: {machine_name}") from exc


def _create_container(client, image, container_name, host_dir, container_dir, ram_size, swap_size):
    options = {
        "name": container_name,
        "tty": True,
        "stdin_open": True,
        "working_dir": container_dir,
        "mem_limit": f"{ram_size}m",
        "memswap_limit": f"{ram_size + swap_size * 1024}m",
        "volumes": {
            str(Path(host_dir).expanduser().resolve()): {
                "bind": container_dir,
                "mode": "rw",
            }
        },
    }
    try:
        return client.containers.create(image, **options)
    except ImageNotFound:
        client.images.pull(image)
        return client.containers.create(image, **options)


def create_container(
    machine_name=DEFAULT_MACHINE_NAME,
    container_name=DEFAULT_CONTAINER_NAME,
    image=DEFAULT_IMAGE,
    host_dir=".",
    container_dir=DEFAULT_CONTAINER_DIR,
    ram_size=None,
    swap_size=None,
):
    if ram_size is None or swap_size is None:
        raise RuntimeError("--ram-size and --swap-size are required")

    with _container_client(machine_name) as client:
        if client.containers.exists(container_name):
            raise RuntimeError(f"Container '{container_name}' already exists.")
        _create_container(
            client,
            image,
            container_name,
            host_dir,
            container_dir,
            ram_size,
            swap_size,
        )
    print(f"Created container '{container_name}'. Start it with: gw container start")


def remove_container(
    machine_name=DEFAULT_MACHINE_NAME,
    container_name=DEFAULT_CONTAINER_NAME,
):
    with _container_client(machine_name) as client:
        if not client.containers.exists(container_name):
            raise RuntimeError(f"Container '{container_name}' not found.")
        client.containers.remove(container_name, force=True)
    print(f"Removed container '{container_name}'")


def start_container(
    machine_name=DEFAULT_MACHINE_NAME,
    container_name=DEFAULT_CONTAINER_NAME,
):
    with _container_client(machine_name) as client:
        client.containers.get(container_name).start()
    print(f"Started container '{container_name}'")


def stop_container(
    machine_name=DEFAULT_MACHINE_NAME,
    container_name=DEFAULT_CONTAINER_NAME,
):
    with _container_client(machine_name) as client:
        client.containers.get(container_name).stop()
    print(f"Stopped container '{container_name}'")


def _response_socket(response):
    candidates = [response.raw]
    for _ in range(8):
        current = candidates[-1]
        for attr in ("_fp", "fp", "raw", "_sock"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate not in candidates:
                candidates.append(candidate)
                break
        else:
            break
    for candidate in candidates:
        if hasattr(candidate, "send") and hasattr(candidate, "recv"):
            return candidate
    raise RuntimeError("Could not attach to container shell")


def _attach_shell(response):
    sock = _response_socket(response)
    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    old_settings = None
    if sys.stdin.isatty():
        old_settings = termios.tcgetattr(stdin_fd)
        tty.setraw(stdin_fd)

    done = threading.Event()

    def read_output():
        try:
            while True:
                chunk = response.raw.read(1024)
                if not chunk:
                    break
                os.write(stdout_fd, chunk)
        finally:
            done.set()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    try:
        while not done.is_set():
            ready, _, _ = select.select([stdin_fd], [], [], 0.2)
            if not ready:
                continue
            data = os.read(stdin_fd, 1024)
            if not data:
                break
            if hasattr(sock, "sendall"):
                sock.sendall(data)
            else:
                sock.send(data)
    except KeyboardInterrupt:
        pass
    finally:
        if old_settings is not None:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
        reader.join(timeout=2)


def open_container(
    machine_name=DEFAULT_MACHINE_NAME,
    container_name=DEFAULT_CONTAINER_NAME,
):
    with _container_client(machine_name) as client:
        container = client.containers.get(container_name)
        container.reload()
        if not container.attrs.get("State", {}).get("Running", False):
            container.start()

        response = container.client.post(
            f"/containers/{container.name}/exec",
            data=json.dumps(
                {
                    "AttachStderr": True,
                    "AttachStdin": True,
                    "AttachStdout": True,
                    "Cmd": ["bash"],
                    "Privileged": False,
                    "Tty": True,
                }
            ),
        )
        response.raise_for_status()
        exec_id = response.json()["Id"]
        response = container.client.post(
            f"/exec/{exec_id}/start",
            data=json.dumps({"Detach": False, "Tty": True}),
            stream=True,
        )
        response.raise_for_status()
        _attach_shell(response)
