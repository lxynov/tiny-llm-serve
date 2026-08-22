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

### Devices

The engine picks `cuda`, then `mps`, then `cpu`, and `--device` overrides it;
CI runs the tests on cpu alone. On Linux `uv sync` resolves the PyPI `torch`
wheel built against CUDA 13.0, so a GPU run there requires CUDA 13.0+. macOS
and Windows pull no CUDA stack.

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

## Benchmark

Every performance claim here is a *delta* between two engine modes measured
under identical conditions, so the harness is fixed before the modes it
compares.

```bash
# Run a seeded workload and write a JSON record to benchmarks/results/
uv run python -m benchmarks.bench_throughput --workload mixed-out --num-requests 8

# Same workload under static batching
uv run python -m benchmarks.bench_throughput --workload mixed-out --num-requests 8 \
  --mode static --batch-size 8
```

`--mode` selects the engine mode, `--dtype bfloat16` suits GPU runs, and
`--device` overrides auto-selection.

| Mode | Engine path | KV reservation counted |
|---|---|---|
| `sequential` | One request at a time, naive growing cache | Exactly what it stores (efficiency 1.0) |
| `static` | Successive full batches of `--batch-size` (all requests are drained, so cross-mode runs stay comparable) | Per wave: `batch × (longest prompt + longest output)`, each slot's preallocated worst case |

In `static` mode `--num-requests` must be a multiple of `--batch-size`. A short
final wave would otherwise be recorded under the full batch size, so a
batch-size sweep would compare trials that never ran the batch size they are
plotted against; requiring whole waves also keeps the workload byte-identical
across the sweep, which is what rule 1 needs.

Continuous batching modes land next and reuse the same flags.

### Sweeps

A batch-size sweep is a grid of runs, not one run, so `benchmarks/sweep.py`
drives it:

```bash
uv run python -m benchmarks.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
  --num-requests 128 --batch-sizes 1,2,4,8,16,32,64,128 --sequential
```

Each trial — one workload at one batch size — runs in **its own process**,
because the caching allocator's pool outlives a run and trials sharing a
process would report each other's peak memory, the metric that decides where
the sweep ends. A trial that dies does not stop the sweep: running out of
memory at a large batch size is *where capacity ran out*, which is a result,
and aborting there would discard every trial still queued behind it. Trials
already recorded in the output directory are skipped, so an interrupted sweep
resumes rather than re-paying.

Batch sizes run ascending behind the optional sequential baseline, so a sweep
that dies at its memory ceiling has already banked the rest of the curve.

A trial that runs out of memory writes a record of its own — `"status": "oom"`,
no metrics — and the sweep then stops raising the batch size for that workload
instead of attempting the rest: memory demand only rises with the batch size,
so every size above one that did not fit would spend a model load apiece to
confirm what the first failure already established. Ceilings are per workload,
so the other curves keep going, and because the ceiling is an ordinary record a
resumed sweep reads it back rather than climbing into it again. Only an
allocation failure counts as one: any other crash is a bug, and reading it as
capacity would silently truncate a curve.

Every trial of one sweep writes into one folder under `--output-dir`, named
after the conditions the sweep holds fixed:

```
benchmarks/results/qwen3-8b-bfloat16-n128-seed0-cuda/
├── sweep.json                                        # the grid, and how each trial ended
├── 20260816-141203-cuda-static-mixed-out-a1b2c3.json
└── ...
```

The name comes from the conditions rather than from the clock because the
folder is what a restart reads: one stamped with the start time would be a new
empty directory every run, and the resume would find nothing to skip. Anything
that makes two trials incomparable — a different model, dtype, request count,
seed, or device — writes somewhere else instead of landing next to results it
cannot be plotted against. `--run-name` overrides the name. `sweep.json` is
rewritten after every trial, so a sweep that dies part way through still leaves
an index, and it carries the one outcome with no record of its own: the batch
sizes the ceiling ruled out before they were ever attempted.

### Workloads

A workload is a seeded list of `(prompt, output_len)` requests generated by
`benchmarks/workloads.py`. Prompts are uniform-random token ids (throughput
does not depend on prompt content); outputs are forced to exact lengths with
`ignore_eos`, so every engine mode processes identical token counts.

| Name | Prompt tokens | Output tokens | What it exposes |
|---|---|---|---|
| `uniform-512x128` | 512 | 128 | Static batching's best case: all sequences finish together |
| `mixed-out` | 512 | log-normal, median 64, clamped [16, 512] | Static batching's tail waste: everyone waits for the longest output |
| `sharegpt-like` | log-normal, median 128, clamped [16, 1024] | same | Long-tailed realistic lengths; preallocation waste |

Records carry `workload_stats` — prompt and output length mean, median, min and
max — because a name and a seed reproduce a workload only while the generator
behind that name is unchanged, and the numbers outlive it.

They also carry `"load": "offline-drain"`: every request exists before the timer
starts and the engine drains them as fast as it can. Nothing arrives, nothing
queues, and request latency is therefore pinned to throughput rather than
independent of it. Load shaped by an arrival rate lands with continuous
batching, and will report fields that mean something different under the same
names — this one keeps the two apart.

### Metrics

| Metric | Definition |
|---|---|
| `wall_time_s` | Wall time of a full workload pass |
| `prefill_time_s` / `decode_time_s` | Wall time in each phase, which together account for the pass. Prefill and decode are different machines — prefill is compute-bound (one large matmul over every prompt token at once), decode is memory-bandwidth-bound (all the weights re-read to produce a single token per sequence) — so a single wall time averages two regimes and hides both. Prefill covers everything up to and including the prompt forward pass, which is where a wave's KV slot pool is allocated; decode covers the step loop and its sampling. Measuring the split costs two extra device barriers per request or wave, not one per step |
| `decode_steps` | Decode forward passes actually run. A wave's first token comes from the prefill logits, so a wave of *n*-token outputs runs *n*−1 of them |
| `s_per_decode_step` | `decode_time_s ÷ decode_steps`. The cleanest diagnostic in a batch-size sweep: roughly flat while host dispatch owns the step, rising once the step is genuinely bandwidth-bound, and the batch size where it turns is where batching stops being free |
| `decode_bytes_read` | Bytes decode must move at minimum: all the weights once per *step* — not once per sequence, which is the batching win stated in bytes — plus each sequence's KV window, which grows by a token every step |
| `mbu` | Memory bandwidth utilization: `decode_bytes_read ÷ decode_time_s` as a fraction of the GPU's peak HBM bandwidth (`benchmarks/hardware.py`). Decode is bandwidth-bound, so this is the fraction of the machine the run is actually using, and `1 − mbu` is the headroom. Null on a device whose peak is not in the table, since a utilization against the wrong ceiling is worse than a missing one |
| `output_tok_s` | Generated tokens ÷ wall time |
| `total_tok_s` | (Prompt + generated tokens) ÷ wall time |
| `weight_bytes` | Bytes of model parameters the device holds. Tied embeddings are one storage under two names, so the vocabulary table is counted once |
| `kv_bytes_reserved` | `kv_efficiency`'s denominator in bytes: reserved KV tokens × (2 × layers × KV heads × head_dim × dtype bytes). Exact by construction, unlike the two peaks below, which the caching allocator reports after the fact — so `weight_bytes + kv_bytes_reserved` predicts a run's floor before it is launched |
| `peak_gpu_memory_allocated_bytes` | `torch.cuda.max_memory_allocated()` over the timed pass (null off-GPU): bytes held by live tensors |
| `peak_gpu_memory_reserved_bytes` | `torch.cuda.max_memory_reserved()` over the same pass: bytes the caching allocator holds from the driver, including freed-but-cached blocks and fragmentation. Always ≥ allocated, and the one OOM is decided by — so it is the number that predicts a sweep's capacity ceiling, and it is what `nvidia-smi` shows minus the CUDA context |
| `peak_concurrent_seqs` | Most sequences in flight at once |
| `kv_efficiency` | KV tokens actually used ÷ KV tokens reserved. Sequential mode's naive cache reserves exactly what it uses (1.0); preallocated batching reserves whole slots up front, and this ratio is the number that indicts it |

### Rules

1. **One variable per comparison.** Same model, dtype, device, commit,
   workload, seed, and sampling (greedy) — only the engine mode changes.
   Never compare runs from different machines or days without a same-day
   rerun of the baseline.
2. **Pre-tokenized prompts.** The engine consumes token ids directly, so
   tokenizer time never pollutes engine numbers.
3. **Bracket the timed pass.** A device barrier
   (`torch.{cuda,mps}.synchronize`) sits on either side of the timer:
   accelerator work is enqueued asynchronously, so an unbracketed timer
   measures kernel *submission*, not execution.
4. **Results are data.** Every run writes one JSON record under
   `benchmarks/results/`; headline numbers must be reproducible from a
   committed record. Each record carries the git commit and whether the tree
   was dirty, plus the environment that produced it — OS, Python, torch, CPU
   model, core count, torch thread count, and GPU name, memory, and CUDA
   version on a GPU box — so rule 1 can be checked after the fact instead of
   taken on trust. A record with `"dirty": true` measures code that no commit
   describes; treat it as a scratch run. Dirty counts tracked modifications
   only — records are written *into* this repository, so counting untracked
   files would mark every run after the first for the record its predecessor
   left behind, and a flag that fires on every run says nothing about any of
   them. Filenames lead with the UTC
   timestamp and device (`20260731-192455-mps-sequential-mixed-out-da20ae`);
   keep separate machines in separate `--output-dir`s.

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

## Site

The project site at <https://lxynov.github.io/tiny-llm-serve/> is built from
`site/` and deploys on every push to `main` that touches it.

```bash
cd site
npm install
npm run dev      # http://localhost:3000/tiny-llm-serve/ — drafts visible, live reload
npm run preview  # http://localhost:4000/tiny-llm-serve/ — exactly what deploys, drafts hidden
npm run build    # writes dist/
npm run new "A post title"

```

See [`site/README.md`](site/README.md).
