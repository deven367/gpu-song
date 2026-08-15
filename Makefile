# gpu-song dev shortcuts
#
#   make sync   install project + dev dependencies (creates .venv as needed)
#   make test   run the pytest suite (coverage gate included)

.DEFAULT_GOAL := sync
.PHONY: sync test

sync:
	uv sync --group dev

test: sync
	uv run pytest
