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
