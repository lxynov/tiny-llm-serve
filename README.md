# tiny-llm-serve

An educational LLM inference engine.

The project is **benchmark-driven**. Its implementation roughly follows this order: static
batching, continuous batching and paged KV cache, so that it reflects the evolution
of inference optimizations in the real world.

Please visit <https://lxynov.github.io/tiny-llm-serve/> for the journey of this project.

## Progress

- [x] Qwen3 model and sampling, validated against Hugging Face
- [ ] Benchmark harness
  - [x] Offline throughput: seeded workloads drained at each batch size of a sweep
  - [x] Decode-step microbenchmark: one engine step at a chosen batch size and KV length
  - [ ] Online: requests arrive at set timestamps, measuring latency (TTFT, TPOT)
- [x] Static batching with a preallocated KV cache, baselined on an H100 ([write-up](https://lxynov.github.io/tiny-llm-serve/posts/basic-static-batching/))
- [x] Improved decoding: fewer host syncs, KV read as a view
- [ ] Further improved decoding: fused decode attention, custom Triton kernel, CUDA graphs
- [ ] Continuous batching
- [ ] Paged KV cache
- [ ] OpenAI-compatible API with streaming
- [ ] Multi-GPU

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync

# Greedy completion with Qwen/Qwen3-0.6B
uv run python -m tiny_llm_serve.engine.llm_engine "Where is Winterfell?"

# With sampling
uv run python -m tiny_llm_serve.engine.llm_engine "Where is Winterfell?" \
  --temperature 0.8 --top-k 50 --top-p 0.95 --max-tokens 64 --seed 0
```

`--help` lists the rest, including `--model` and `--stop`. The engine runs on
CUDA, MPS or CPU, picked in that order. On Linux, `uv sync` installs PyTorch
built against CUDA 13.0, so a GPU there needs CUDA 13.0+.

## Benchmarks

Every performance claim is a delta between two runs that differ in one thing
only. Each run writes a JSON record, and the records are committed under
[`benchmarks/results/`](benchmarks/results/), so any number I quote can be
traced back to one.

```bash
# Throughput at batch sizes 1 to 128, draining every workload at each
uv run python -m benchmarks.throughput.sweep --model Qwen/Qwen3-8B

# The cost of a single decode step, at a few (batch, kv_len) cells
uv run python -m benchmarks.decode_step.bench --model Qwen/Qwen3-8B
```

- [docs/benchmarks.md](docs/benchmarks.md): how each benchmark runs, and the
  rules that keep runs comparable
- [docs/metrics.md](docs/metrics.md): the workloads, and what each number
  means

## Development

```bash
uv run pytest        # tests
uv run ruff check    # lint
uv run ruff format   # format
uv run pyrefly check # type check
uv build             # sdist and wheel
```

The tests run against the real `Qwen/Qwen3-0.6B` checkpoint, which the first
`uv run pytest` downloads into the Hugging Face cache (~1.4 GB). They skip on a
machine that is both uncached and offline. CI runs them on CPU.

The site is built from [`site/`](site/README.md) and deploys on every push to
`main` that touches it.
