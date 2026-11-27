export PATH := $(HOME)/.local/bin:$(PATH)

VERSION  = $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
TAG      = v$(VERSION)
ARTIFACT = goodwoolf-$(TAG).zip
REPO ?= $(shell git remote get-url origin | sed -E 's#^(git@|ssh://git@|https://)[^:/]+[:/]##; s#\.git$$##')

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

.PHONY: check-gh
check-gh:
	@command -v gh >/dev/null 2>&1 || \
		(echo "gh CLI not found." && exit 1)
	@gh auth status >/dev/null 2>&1 || \
		(echo "gh is not authenticated. Run 'gh auth login' or set GH_TOKEN." && exit 1)

.PHONY: release
release: check-gh build
	@test -f RELEASE.md || (echo "RELEASE.md not found." && exit 1)
	@test -z "$$(git status --porcelain)" || \
		{ echo "Working tree has uncommitted changes. Commit them first."; exit 1; }
	(cd dist && zip -rq ../$(ARTIFACT) .)
	@git rev-parse "$(TAG)" >/dev/null 2>&1 || git tag $(TAG)
	git push origin $(TAG)
	gh release create $(TAG) $(ARTIFACT) --repo $(REPO) --verify-tag --title "$(TAG)" --notes-file RELEASE.md

.PHONY: clean
clean:
	rm -f $(ARTIFACT)
	rm -rf dist/ build/ *.egg-info/ src/*.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ .tox/ .nox/ htmlcov/ .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete
