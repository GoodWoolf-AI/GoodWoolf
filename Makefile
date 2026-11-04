# Setup target
.PHONY: setup
setup:
	uv sync

# Build target
.PHONY: build
build:
	uv build

# Publish target
.PHONY: publish
publish:
	uv publish

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
