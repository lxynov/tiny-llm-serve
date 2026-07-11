# tiny-llm-serve

A tiny LLM inference and serving engine, built from scratch in Python + PyTorch.

## Setup

```bash
# Install uv (macOS/Linux) if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv (fetching Python 3.12 if needed) and install all dependencies
uv sync
```

Commands below run inside `.venv` via `uv run`. To use the environment
directly, run `source .venv/bin/activate`.

## Development

```bash
uv run pytest        # run tests
uv run ruff check    # lint
uv run ruff format   # format
```

## Build

```bash
uv build
```
