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

## Run

```bash
# Greedy (temperature 0) completion of the default prompt
uv run python -m tiny_llm_serve.engine.llm_engine "Where is Winterfell?"

# Sampling: temperature, top-k, top-p, repetition penalty, seed
uv run python -m tiny_llm_serve.engine.llm_engine "Where is Winterfell?" \
  --temperature 0.8 --top-k 50 --top-p 0.95 \
  --repetition-penalty 1.1 --max-tokens 64 --seed 0

# Stop as soon as a string appears (repeat --stop for more than one)
uv run python -m tiny_llm_serve.engine.llm_engine "Count: 1 2 3" --stop $'\n' --stop "10"
```

`--model` selects the checkpoint (default `Qwen/Qwen3-0.6B`), and the device is
picked automatically.

## Development

```bash
uv run pytest        # run tests
uv run ruff check    # lint
uv run ruff format   # format
uv run pyrefly check # type check
```

The tests check against the real `Qwen/Qwen3-0.6B` checkpoint, which the first
`uv run pytest` downloads into the Hugging Face cache (~1.4GB). Later runs reuse
it. Tests that need it skip on a machine that is both uncached and offline.

## Build

```bash
uv build
```
