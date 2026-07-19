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

The tests check against the real `Qwen/Qwen3-0.6B` checkpoint, which the first
`uv run pytest` downloads into the Hugging Face cache (~1.4GB). Later runs reuse
it. Tests that need it skip on a machine that is both uncached and offline.

## Build

```bash
uv build
```
