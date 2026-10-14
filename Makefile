.PHONY: setup check test smoke evaluate collect wipe compare

setup:
	uv sync --extra dev

check:
	uv run ruff check .

test:
	uv run pytest

smoke:
	uv run sim2skill smoke

evaluate:
	uv run sim2skill evaluate --episodes 20

collect:
	uv run sim2skill collect --episodes 20 --output artifacts/demos.npz

wipe:
	uv run sim2skill evaluate --task wipe --policy oracle --episodes 8

compare:
	uv run sim2skill compare --episodes 8 --train-episodes 16
