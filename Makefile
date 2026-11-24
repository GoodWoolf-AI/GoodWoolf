# Setup Python environment
.PHONY: setup
setup:
	uv sync

# Install publish dependencies (twine, wheel)
.PHONY: setup-publish
setup-publish:
	uv pip install ".[publish]"

# Build distribution artifacts
.PHONY: build
build:
	uv build

# Publish to PyPi.org
.PHONY: publish
publish:
	uv publish

# Publish to test.PyPi.org
.PHONY: publish-test
publish-test:
	uv run python -m twine upload --repository testpypi dist/*

# Install editable
.PHONY: install-edit
install-edit:
	uv pip install -e .

# Uninstall editable
.PHONY: uninstall-edit
uninstall-edit:
	uv pip uninstall .

# Install from dist (non-editable)
.PHONY: install-dist
install-dist:
	uv pip install dist/goodwoolf-*.whl

# Uninstall from dist
.PHONY: uninstall-dist
uninstall-dist:
	uv pip uninstall goodwoolf

# Podman machine specs
MACHINE_NAME ?= goodwoolf-machine
CPUS         ?= 2
DISK_SIZE    ?=
RAM_SIZE     ?=
SWAP_SIZE    ?=

# Container specs
CONTAINER_NAME ?= goodwoolf
IMAGE          ?= docker.io/library/ubuntu:24.04

# Shared folder: HOST_DIR on your machine appears as CONTAINER_DIR in the container
HOST_DIR       ?= $(CURDIR)
CONTAINER_DIR  ?= /workspace

# Where gvproxy lives (found via PATH, no hardcoded dirs)
GVPROXY      := $(shell command -v gvproxy 2>/dev/null)
# virtiofsd: on PATH, or in /usr/libexec (where Debian/Fedora install it)
VIRTIOFSD    := $(or $(shell command -v virtiofsd 2>/dev/null),$(wildcard /usr/libexec/virtiofsd))

# Generated per-project podman config that points podman at gvproxy's directory.
# Podman only searches its own helper dirs, not PATH, so this is required
# when gvproxy is in e.g. /usr/bin.
PODMAN_CONF  := $(CURDIR)/.podman-helpers.conf
PODMAN_ENV   := PATH="$(dir $(VIRTIOFSD)):$$PATH" CONTAINERS_CONF_OVERRIDE=$(PODMAN_CONF)

# Swap setup script, run as root inside the podman machine
define SWAP_SCRIPT
set -euo pipefail
if swapon --show=NAME --noheadings | grep -q /var/swapfile; then
    echo "Swap already active on /var/swapfile"
    swapon --show
    exit 0
fi
FS_TYPE=$$(df -T /var | awk 'NR==2 {print $$2}')
echo "Detected /var filesystem: $$FS_TYPE"
if [ ! -f /var/swapfile ]; then
    if [ "$$FS_TYPE" = "btrfs" ]; then
        btrfs filesystem mkswapfile --size $(SWAP_SIZE)g /var/swapfile
    else
        fallocate -l $(SWAP_SIZE)G /var/swapfile
        chmod 600 /var/swapfile
        mkswap /var/swapfile
    fi
fi
swapon /var/swapfile
grep -q '/var/swapfile' /etc/fstab || echo '/var/swapfile none swap sw 0 0' >> /etc/fstab
swapon --show
endef
export SWAP_SCRIPT

# Check podman and gvproxy are installed
.PHONY: check-machine
check-machine:
	@command -v podman >/dev/null 2>&1 || { \
		echo "Error: podman is not installed."; \
		echo "Install it, e.g.: sudo apt install podman"; \
		exit 1; \
	}
	@[ -n "$(GVPROXY)" ] || { \
		echo "Error: gvproxy not found on PATH."; \
		echo "Install it: sudo apt install gvproxy"; \
		echo "or download gvproxy-linux-amd64 from https://github.com/containers/gvisor-tap-vsock/releases"; \
		echo "and install it: sudo install -m 755 gvproxy-linux-amd64 /usr/local/bin/gvproxy"; \
		exit 1; \
	}
	@[ -n "$(VIRTIOFSD)" ] || { \
		echo "Error: virtiofsd not found (checked PATH and /usr/libexec)."; \
		echo "Install it: sudo apt install virtiofsd"; \
		exit 1; \
	}
	@echo "podman:    $$(command -v podman)"
	@echo "gvproxy:   $(GVPROXY)"
	@echo "virtiofsd: $(VIRTIOFSD)"

# Write podman config override pointing at gvproxy's directory
$(PODMAN_CONF): check-machine
	@printf '[engine]\nhelper_binaries_dir = ["%s"]\n' "$(dir $(GVPROXY))" > $@
	@echo "Wrote $@ (helper_binaries_dir=$(dir $(GVPROXY)))"

# Create Podman machine
.PHONY: setup-machine
setup-machine: check-machine $(PODMAN_CONF)
ifeq ($(strip $(RAM_SIZE)),)
	$(error RAM_SIZE is not set. Usage: make setup-machine DISK_SIZE=100 SWAP_SIZE=50 RAM_SIZE=8192)
endif
ifeq ($(strip $(DISK_SIZE)),)
	$(error DISK_SIZE is not set. Usage: make setup-machine DISK_SIZE=100 SWAP_SIZE=50 RAM_SIZE=8192)
endif
ifeq ($(strip $(SWAP_SIZE)),)
	$(error SWAP_SIZE is not set. Usage: make setup-machine DISK_SIZE=100 SWAP_SIZE=50 RAM_SIZE=8192)
endif
	@if $(PODMAN_ENV) podman machine inspect $(MACHINE_NAME) >/dev/null 2>&1; then \
		echo "Error: Podman machine '$(MACHINE_NAME)' already exists."; \
		echo "Remove it first with: make clear-machine"; \
		echo "or use another name:  make setup-machine MACHINE_NAME=other ..."; \
		exit 1; \
	fi
	@echo "1. Initializing Podman machine '$(MACHINE_NAME)' (RAM: $(RAM_SIZE) MiB, Disk: $(DISK_SIZE) GiB)..."
	# To add HOST_DIR to the machine use volume option
	# --volume $(HOST_DIR):$(HOST_DIR) \
	@$(PODMAN_ENV) podman machine init \
		--cpus $(CPUS) \
		--memory $(RAM_SIZE) \
		--swap 0 \
		--disk-size $(DISK_SIZE) \
		$(MACHINE_NAME)

	@echo "2. Starting '$(MACHINE_NAME)'..."
	@state=$$($(PODMAN_ENV) podman machine inspect --format '{{.State}}' $(MACHINE_NAME)); \
	if [ "$$state" = "running" ]; then \
		echo "Machine '$(MACHINE_NAME)' already running."; \
	else \
		$(PODMAN_ENV) podman machine start $(MACHINE_NAME); \
	fi

	@echo "3. Creating and enabling $(SWAP_SIZE)G disk swap inside '$(MACHINE_NAME)'..."
	@printf '%s\n' "$$SWAP_SCRIPT" | $(PODMAN_ENV) podman machine ssh $(MACHINE_NAME) "sudo bash -s"
	@echo "Done! Machine '$(MACHINE_NAME)' is ready with $(RAM_SIZE) MiB RAM and $(SWAP_SIZE) GiB swap."

# Open an SSH shell in the Podman machine (VM), starting it if needed
.PHONY: open-machine
open-machine:
	@state=$$($(PODMAN_ENV) podman machine inspect --format '{{.State}}' $(MACHINE_NAME)); \
	if [ "$$state" != "running" ]; then \
		$(PODMAN_ENV) podman machine start $(MACHINE_NAME); \
	fi
	@$(PODMAN_ENV) podman machine ssh $(MACHINE_NAME)

# Remove the Podman machine and generated config
.PHONY: clear-machine
clear-machine:
	-$(PODMAN_ENV) podman machine rm -f $(MACHINE_NAME)
	-rm -f $(PODMAN_CONF)

# Create a container in the Podman machine with memory/swap limits
# (--memory-swap is memory + swap combined, in MiB)
.PHONY: create-container
create-container:
ifeq ($(strip $(RAM_SIZE)),)
	$(error RAM_SIZE is not set. Usage: make create-container DISK_SIZE=60 SWAP_SIZE=50 RAM_SIZE=8192)
endif
ifeq ($(strip $(SWAP_SIZE)),)
	$(error SWAP_SIZE is not set. Usage: make create-container DISK_SIZE=60 SWAP_SIZE=50 RAM_SIZE=8192)
endif
	@if podman --connection $(MACHINE_NAME) container exists $(CONTAINER_NAME); then \
		echo "Error: container '$(CONTAINER_NAME)' already exists."; \
		echo "Remove it first with: make remove-container"; \
		exit 1; \
	fi
	@echo "Creating container '$(CONTAINER_NAME)' from $(IMAGE) (RAM: $(RAM_SIZE) MiB, Swap: $(SWAP_SIZE) GiB)..."
	@podman --connection $(MACHINE_NAME) create -it \
		--name $(CONTAINER_NAME) \
		--memory $(RAM_SIZE)m \
		--memory-swap $$(( $(RAM_SIZE) + $(SWAP_SIZE) * 1024 ))m \
		--volume $(HOST_DIR):$(CONTAINER_DIR) \
		--workdir $(CONTAINER_DIR) \
		$(IMAGE)
	@echo "Created. Start it with: make start-container, open a shell with: make open-container"

# Remove the container
.PHONY: remove-container
remove-container:
	@podman --connection $(MACHINE_NAME) rm -f $(CONTAINER_NAME)

# Start the container in the background (stays alive because it has a TTY)
.PHONY: start-container
start-container:
	@podman --connection $(MACHINE_NAME) start $(CONTAINER_NAME)

# Stop the container
.PHONY: stop-container
stop-container:
	@podman --connection $(MACHINE_NAME) stop $(CONTAINER_NAME)

# Open an interactive bash session in the container (starts it if needed).
# Run it in several terminals to get several sessions; exiting one doesn't stop the container.
.PHONY: open-container
open-container: start-container
	@podman --connection $(MACHINE_NAME) exec -it $(CONTAINER_NAME) bash
