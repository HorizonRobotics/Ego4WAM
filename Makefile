# Source provenance: StarVLA project (https://github.com/starVLA/starVLA).
# The upstream file has no file-level copyright notice; none is inferred here.
# SPDX-License-Identifier: MIT
# Repository-level license: LICENSES/StarVLA-MIT.txt.

.PHONY: help clean check autoformat
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Apply formatting in place with black and fixable Ruff edits without failing on existing lint backlog."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix-only --show-fixes .
