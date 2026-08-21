# Static Batching on an H100

A measured baseline for `tiny-llm-serve`: what static batching buys, what it
wastes, and where the engine's own implementation becomes the limit. Every
number below comes from a committed record under
[`benchmarks/results/qwen3-8b-bfloat16-n512-seed0-cuda/`](../benchmarks/results/qwen3-8b-bfloat16-n512-seed0-cuda),
and is reproducible with `python -m benchmarks.report` on that folder.

**The three findings that matter.** Batching works, and stops working sooner
than the batch size suggests: throughput saturates around batch 64–128 and the
last doubling to 256 buys 14%. On a realistic length distribution, **81% of
decode slots are advancing sequences that have already finished** — the waste
continuous batching exists to remove. And underneath both, decode never exceeds
**25% of the H100's memory bandwidth at any batch size**, which is a ceiling
that belongs to eager PyTorch rather than to the batching strategy.

---

## 1. Implementation

### Scope

An inference engine written from scratch in Python and PyTorch — no vLLM, no
TensorRT-LLM, no FlashInfer, no custom CUDA. The model, the KV cache, the
attention path, the sampler and the scheduler are all in-repo: 1,350 lines of
engine under `src/`, and 1,347 more of benchmark tooling. It runs on CUDA, MPS
and CPU from the same code.

That constraint is the point: every layer of the stack is inspectable, so a
measurement can be traced to the line that caused it. It is also why the memory
bandwidth result below is interesting rather than embarrassing — the gap
between this engine and a production one is exactly what the remaining work is.

### Model

[`models/qwen3.py`](../src/tiny_llm_serve/models/qwen3.py) implements the Qwen3
decoder:

| Component | Implementation |
|---|---|
| Attention | `F.scaled_dot_product_attention` with `enable_gqa=True` — dispatches to the FlashAttention kernel on CUDA, math fallback on CPU |
| Projections | QKV merged into one `MergedLinear`, gate/up merged likewise — one matmul instead of three, and the checkpoint is sharded into them at load |
| Normalization | RMSNorm, plus Qwen3's per-head q/k RMSNorm applied *before* RoPE |
| Activation | SwiGLU (`SiluAndMul`) |
| Positions | Rotary embeddings from a precomputed cos/sin cache |

For `Qwen/Qwen3-8B` in bfloat16 that is 16.4 GB of weights, and its 36 layers ×
8 KV heads × 128 head_dim give **144 KiB of KV per token** — the constant behind
every capacity number later. Both figures come out of the records themselves
(`weight_bytes`, and `kv_bytes_reserved` divided by the tokens a wave reserved)
rather than being assumed.

### KV cache: two backends, one protocol

[`kv/base.py`](../src/tiny_llm_serve/kv/base.py) splits the cache into two
contracts, which is what lets the engine change batching strategy without the
model noticing:

- **`KVCacheView`** — the per-step surface a decoder layer sees: somewhere to
  `append` this step's k/v, and the `attn_mask` describing which cached
  positions are real. A layer only ever appends to whatever view it is handed.
- **`KVManager`** — the per-request surface the engine sees: `can_admit`,
  `admit`, `free`, `begin_prefill`, `begin_decode`. Admission policy lives
  here and nowhere else.

Two implementations sit behind them. The **naive** cache grows with exactly what
it stores and backs sequential mode. The **preallocated** cache
([`kv/preallocated.py`](../src/tiny_llm_serve/kv/preallocated.py)) is a fixed
pool of `num_slots` slots, each reserving `max_model_len` contiguous tokens per
layer, and backs static mode. Its reservation is what makes an admitted
sequence safe by construction — it can never outgrow its slot mid-flight — and
it is also the thing that wastes memory.

The two masking regimes are worth stating, because they are where batched
attention actually gets decided:

- **Prefill** writes a right-padded prompt block and passes `attn_mask=None`.
  Causal masking alone already keeps every real query away from pad keys, and
  the pad rows' outputs are discarded.
- **Decode** appends one token per row at each row's own current length, and
  builds an explicit `[batch, 1, 1, kv_len]` validity mask. That mask hides both
  the stale pad k/v left behind by prefill and the tails of rows shorter than
  the longest sequence in the step.

### The static batching loop

`LLM.generate_batch_ids` in
[`engine/llm_engine.py`](../src/tiny_llm_serve/engine/llm_engine.py) is the
Orca/FasterTransformer-style baseline: **one padded prefill for the whole wave,
then lockstep decode until every sequence in it is done.**

Three properties define it, and all three are what the results measure:

1. **Nothing is admitted mid-flight.** A wave is fixed at prefill.
2. **A finished sequence idles in its slot.** It is still stepped every
   iteration — fed its own already-seen first token so the step stays
   well-formed for any vocabulary — and its output is discarded.
3. **The reservation is sized for the worst case in the wave:**
   `max_model_len = max(prompt_len) + max(max_tokens)`. Not each sequence's
   own need, because an early finisher keeps its slot and keeps writing KV
   until the wave drains.

Points 2 and 3 are the same defect seen from the compute side and the memory
side, and §2 measures them separately.

### Benchmark harness

Four modules under [`benchmarks/`](../benchmarks), each with one job:

| Module | Job |
|---|---|
| `workloads.py` | Seeded `(prompt, output_len)` request lists. Prompts are uniform-random token ids — throughput doesn't depend on prompt content — and outputs are forced to exact lengths with `ignore_eos`, so every engine mode processes identical token counts |
| `bench_throughput.py` | One trial → one JSON record, carrying metrics, workload stats, git commit, dirty flag and full environment |
| `sweep.py` | Drives a workload × batch-size grid. Each trial gets **its own process** (the caching allocator's pool outlives a run, so trials sharing one would report each other's peak memory), a failed trial does not end the sweep, and an out-of-memory result is recorded as the workload's ceiling rather than retried upward |
| `report.py` | Joins the index to the records, derives the columns none of them carry, and checks that the folder is actually one comparison |

The measurement protocol is four rules, stated in full in the
[README](../README.md#rules): one variable per comparison; pre-tokenized
prompts so tokenizer time never pollutes engine numbers; a device barrier on
either side of the timer, since accelerator work is enqueued asynchronously and
an unbracketed timer measures kernel *submission*; and every headline number
reproducible from a committed record.

### Tech stack

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.12 |
| Compute | PyTorch | 2.13.0+cu130 (CUDA 13.0) |
| Weight loading | safetensors | 0.8.0 |
| Checkpoint fetch | huggingface-hub | 1.24.0 |
| Tokenizer | transformers | 5.14.1 |
| Packaging | uv / `uv_build` | — |
| Tests | pytest | 9.1.1 |
| Lint / format | ruff | 0.15.21 |
| Types | pyrefly | 1.1.1 |
| Plots (optional) | matplotlib | 3.11.1 |

Four runtime dependencies, and only one of them touches the forward pass.
`transformers` supplies `AutoTokenizer` and nothing else — weights go from
safetensors straight into the from-scratch module tree, so no upstream modeling
code runs. `matplotlib` is a dev dependency imported lazily by
`report.py --plot`; a report printed to a terminal never loads it.

---

## 2. Results

### Conditions

| | |
|---|---|
| Model | `Qwen/Qwen3-8B`, bfloat16 |
| Hardware | 1× NVIDIA H100 80GB HBM3 (SXM5), Intel Xeon Platinum 8480+ |
| Software | torch 2.13.0+cu130, CUDA 13.0, Python 3.12.3, Linux 6.8 |
| Workload | 512 requests, seed 0, all sequences drained offline |
| Grid | 3 workloads × 10 batch sizes; 26 of 30 trials completed |
| Commit | `faa672214e4f`, clean tree on every trial |

The three workloads probe different things. `uniform-512x128` is static
batching's best case — every sequence finishes on the same step. `mixed-out`
holds prompts fixed at 512 and gives outputs a log-normal tail. `sharegpt-like`
makes both lengths long-tailed, approximating a real trace.

Column notes: **`occ`** is the fraction of decode slots holding a sequence that
is still generating, so `1 − occ` is compute spent on finished sequences.
**`kv_eff`** is KV tokens used ÷ KV tokens reserved. **`mbu`** is memory
bandwidth utilization against the H100's published 3.35 TB/s.

### uniform-512x128 — the best case

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 1539.5 | 0.9% | 23.5 | 42.6 | 1.0 | 1.000 | 1.000 | 0.209 | 15.5 |
| 2 | ok | 758.5 | 1.4% | 23.0 | 86.4 | 2.0 | 1.000 | 1.000 | 0.215 | 15.7 |
| 4 | ok | 393.4 | 2.4% | 23.6 | 166.6 | 3.9 | 1.000 | 1.000 | 0.211 | 16.0 |
| 8 | ok | 208.1 | 4.3% | 24.5 | 315.0 | 7.4 | 1.000 | 1.000 | 0.208 | 16.6 |
| 16 | ok | 122.4 | 7.2% | 28.0 | 535.5 | 12.6 | 1.000 | 1.000 | 0.189 | 18.0 |
| 32 | ok | 63.8 | 13.7% | 27.1 | 1026.6 | 24.1 | 1.000 | 1.000 | 0.210 | 20.7 |
| 64 | ok | 38.2 | 22.9% | 29.0 | 1714.3 | 40.3 | 1.000 | 1.000 | 0.225 | 26.2 |
| 128 | ok | 28.4 | 32.1% | 38.0 | 2305.0 | 54.1 | 1.000 | 1.000 | 0.214 | 37.0 |
| 256 | ok | 24.8 | 35.9% | 62.6 | 2638.5 | 62.0 | 1.000 | 1.000 | 0.182 | 58.8 |
| 512 | oom | - | - | - | - | - | - | - | - | - |

### mixed-out — a long-tailed output distribution

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 1229.1 | 1.0% | 23.8 | 42.0 | 1.0 | 1.000 | 1.000 | 0.207 | 15.5 |
| 2 | ok | 922.1 | 1.2% | 23.7 | 56.0 | 1.3 | 0.664 | 0.924 | 0.209 | 15.8 |
| 4 | ok | 649.8 | 1.5% | 23.9 | 79.4 | 1.9 | 0.477 | 0.849 | 0.209 | 16.2 |
| 8 | ok | 448.1 | 2.0% | 24.5 | 115.2 | 2.7 | 0.357 | 0.773 | 0.209 | 17.3 |
| 16 | ok | 315.8 | 2.8% | 25.5 | 163.5 | 3.9 | 0.265 | 0.689 | 0.212 | 21.0 |
| 32 | ok | 199.9 | 4.4% | 26.3 | 258.3 | 6.1 | 0.220 | 0.633 | 0.226 | 26.7 |
| 64 | ok | 123.9 | 7.1% | 29.5 | 416.8 | 9.9 | 0.205 | 0.612 | 0.238 | 29.5 |
| 128 | ok | 91.7 | 9.6% | 40.6 | 562.9 | 13.4 | 0.195 | 0.598 | 0.227 | 43.8 |
| 256 | ok | 78.5 | 11.2% | 68.2 | 657.9 | 15.7 | 0.195 | 0.598 | 0.199 | 72.3 |
| 512 | oom | - | - | - | - | - | - | - | - | - |

### sharegpt-like — long-tailed prompts *and* outputs

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 2372.7 | 1.0% | 24.1 | 41.2 | 1.0 | 1.000 | 1.000 | 0.203 | 15.8 |
| 2 | ok | 1778.1 | 0.9% | 24.4 | 55.0 | 1.3 | 0.675 | 0.673 | 0.202 | 16.4 |
| 4 | ok | 1329.5 | 0.9% | 25.2 | 73.5 | 1.8 | 0.465 | 0.471 | 0.199 | 17.4 |
| 8 | ok | 892.6 | 1.3% | 25.6 | 109.5 | 2.7 | 0.354 | 0.355 | 0.203 | 22.5 |
| 16 | ok | 623.1 | 2.0% | 26.6 | 156.9 | 3.8 | 0.265 | 0.269 | 0.213 | 25.4 |
| 32 | ok | 411.3 | 3.5% | 28.9 | 237.7 | 5.8 | 0.221 | 0.222 | 0.232 | 29.5 |
| 64 | ok | 284.8 | 5.6% | 36.3 | 343.3 | 8.3 | 0.205 | 0.201 | 0.245 | 43.8 |
| 128 | ok | 247.8 | 6.8% | 56.4 | 394.5 | 9.6 | 0.186 | 0.185 | 0.238 | 72.3 |
| 256 | oom | - | - | - | - | - | - | - | - | - |
| 512 | not attempted | - | - | - | - | - | - | - | - | - |

### Finding 1 — batching works, then abruptly stops working

| Workload | bs=1 | Peak | at bs | Speedup | % of linear | Last doubling |
|---|---|---|---|---|---|---|
| `uniform-512x128` | 42.6 | **2,638** tok/s | 256 | 62× | 24% | +14% |
| `mixed-out` | 42.0 | **658** tok/s | 256 | 16× | 6% | +17% |
| `sharegpt-like` | 41.2 | **394** tok/s | 128 | 10× | 7% | +15% |

Scaling holds near-linear to batch 32 on the uniform workload (24× for 32×),
then bends hard. Past 128, every doubling costs roughly 60% more memory for
under 20% more throughput. **The useful operating point is batch 64–128 on all
three workloads**, well below the memory ceiling — the ceiling is not what
limits you first.

### Finding 2 — decode is memory-bound, and you can watch it stop being so

Per-step decode time on `uniform-512x128`: 23.5 ms at batch 1, 29.0 ms at batch
**64**. Sixty-four sequences for 1.23× the time — the batch rides along free
because the step is dominated by re-reading 16.4 GB of weights, not by the
arithmetic. Then 38.0 ms at 128 and 62.6 ms at 256, roughly doubling with the
batch: the step has become compute- and KV-bound.

`s_per_decode_step` is the cleanest diagnostic in the whole sweep. Flat means
the batch is free; the batch size where it turns is where batching stops paying,
and it lands at 64–128, matching Finding 1 from a different direction.

### Finding 3 — the engine, not the strategy, is the ceiling

Effective decode bandwidth never leaves a narrow band:

| Workload | Effective bandwidth | MBU |
|---|---|---|
| `uniform-512x128` | 609–752 GB/s | 0.18–0.22 |
| `mixed-out` | 665–799 GB/s | 0.20–0.24 |
| `sharegpt-like` | 666–821 GB/s | 0.20–0.25 |

**Against the H100's 3.35 TB/s, that is 18–25% of the machine, at every batch
size and on every workload.** Batching never moves it.

At batch 1 the reading is at its cleanest: one decode step reads 16.4 GB of
weights in 23.5 ms — 697 GB/s. Nothing about batching, scheduling or padding is
involved. That number is the eager-PyTorch execution overhead by itself: ~36
layers' worth of un-fused kernel launches, per token, with the host in the loop
every step. **There is a 4–5× speedup available in the implementation before any
scheduling change is worth making** — and it would multiply with, not overlap,
everything in §3.

### Finding 4 — on realistic lengths, four decode slots in five are wasted

At each workload's best batch size:

| Workload | `occ` | Decode spent on finished sequences |
|---|---|---|
| `uniform-512x128` | 1.00 | 0% |
| `mixed-out` | 0.20 | **80%** |
| `sharegpt-like` | 0.19 | **81%** |

This is the whole indictment of static batching in one number, and it accounts
for most of the 6.7× throughput gap between the best case (2,638 tok/s) and the
realistic one (394 tok/s): dividing `sharegpt-like`'s 394 tok/s by its 0.19
occupancy gives ~2,100 tok/s, about 80% of the way to the uniform workload. The
waves are running to their
*longest* member: on `sharegpt-like` at batch 128, 4,092 decode steps for a
workload whose average output is 191 tokens.

Note the shape of the `occ` column — it falls steeply from batch 2 to 32 and
then flattens near 0.19. That floor is a property of the length distribution,
not of the batch size: past about 64 sequences, a wave almost surely contains
one near-maximum output, and everything else waits for it. **Larger batches
cannot fix this, and the flattening is why.**

### Finding 5 — `occ` and `kv_eff` measure different waste, and diverge

On `mixed-out` at batch 256, `kv_eff` reads 0.60 while `occ` reads 0.20. The
memory looks two-thirds used while four decode slots in five are advancing a
sequence that has already stopped.

The cause is that the reservation is `max(prompt) + max(output)` and this
workload's prompts are a fixed 512 tokens. The prompt half of every slot is
genuinely used, so it dilutes the ratio; the output half is mostly padding. On
`sharegpt-like`, where nothing is fixed, the two converge (0.185 vs 0.186).

The practical consequence: **`kv_efficiency` alone understates the problem on
any workload with uniform prompts**, and a scheduler tuned against it would stop
optimizing too early. Both columns are needed.

### Finding 6 — the memory ceiling is set by the tail, not the mean

| Workload | Ceiling | Last successful batch | Peak reserved |
|---|---|---|---|
| `uniform-512x128` | 512 | 256 | 58.8 GB |
| `mixed-out` | 512 | 256 | 72.3 GB |
| `sharegpt-like` | 256 | 128 | 72.3 GB |

`sharegpt-like` dies a full doubling earlier than the others, at half the batch
size, because each wave reserves `batch × (longest prompt + longest output)` —
and with both distributions long-tailed, the longest prompt and the longest
output in a wave — each clamped at 1,024 tokens, and rarely the same request —
together set the slot size for all 128 sequences. The ceiling tracks the tail of the distribution,
not its mean, which is exactly the pathology block-based allocation removes.

### Finding 7 — prefill becomes the next ceiling

Prefill is a near-constant ~9 s on `uniform-512x128` regardless of batch size —
it is the same 262,144 prompt tokens either way. But as decode gets faster it
goes from 0.9% of wall time at batch 1 to **36% at batch 256**.

Amdahl's law is already binding on the best-case workload: even an infinitely
fast decode would only cut that run from 24.8 s to 8.9 s. Decode-side work has
a visible ceiling here, and prefill has had no optimization attention at all.

### Caveats

- **n = 1 per cell.** No repeats, so no variance estimate. One cell looks off:
  `uniform-512x128` at batch 16 runs 28.0 ms/step, *slower* than batch 32's
  27.1, with its neighbors at 24.5 and 27.1. Probably noise, but a single
  sample cannot say.
- **`decode_bytes_read` is a model, not a measurement.** It is the minimum
  bytes decode must move, derived from weight and KV sizes. MBU inherits its
  assumptions, and it deliberately charges weights once per *padded* step — so
  MBU measures bytes moved, not bytes usefully moved, and is blind to Finding
  4. That is why a flat MBU column sits next to a 6.7× goodput spread.
- **Offline-drain load only.** Every request exists before the timer starts.
  There are no arrivals, no queueing, and therefore no meaningful latency
  numbers — TTFT and inter-token latency here would be artifacts of the drain
  order, not service quality.
- **Random-token prompts.** Fine for throughput, which doesn't depend on prompt
  content, but it means nothing here validates output quality.
- **No sequential baseline in this folder.** Batch 1 under static batching
  stands in for it, which is close but not identical to the naive growing cache.

---

## 3. Future work

Ordered by what this data says the payoff is.

### Continuous batching — the headline fix

Findings 4 and 6 are both symptoms of admitting work only at wave boundaries.
Replacing the wave loop with a scheduler that admits a new sequence into a slot
the moment its occupant finishes attacks both: `occ` should approach 1.0, and
the ceiling should stop tracking the distribution's tail.

The prediction worth writing down before the run: **`sharegpt-like` throughput
should move from 394 tok/s toward the uniform workload's curve**, since the two
differ almost entirely by padding. If it lands near 2,000 tok/s the mechanism is
confirmed; if it stalls around 800, something else is binding and Finding 3 is
the first suspect. `mixed-out` and `sharegpt-like` at batch 64–128 are the cells
to run first.

The harness is already built for this comparison — same workloads, same seed,
same records, one variable changed.

### Paged attention

Continuous batching still leaves each slot reserving `max_model_len`
contiguously. Block-based allocation would decouple reservation from the
distribution's tail, which is what Finding 6 is really about. `KVManager`
already anticipates this: the docstring notes a paged backend claims blocks
incrementally behind the same interface, so it is a backend swap rather than an
engine change.

### Close the 4–5× implementation gap

Finding 3 is orthogonal to everything above and probably the largest single
multiplier in this document. In rough order of expected return:

- **CUDA graphs** for the decode step. At 23.5 ms/step with ~36 layers of
  un-fused kernels, per-step host dispatch is the prime suspect, and capturing
  the step removes it wholesale.
- **`torch.compile`** on the decoder layer, to fuse the RMSNorm/RoPE/SwiGLU
  chain that currently runs as separate kernels.
- **Profile before either.** The MBU number says ~80% of the machine is idle
  but not *why*; an Nsight or `torch.profiler` trace of one decode step would
  distinguish launch overhead from something less convenient. This should come
  first — it is cheap, and it decides the order of the other two.

### Prefill

Untouched so far, and Finding 7 says it is already 36% of wall time at high
batch. Chunked prefill — slicing long prompts and interleaving them with decode
steps — would keep prefill from monopolizing the device, and matters more once
continuous batching makes admission continuous.

### Measurement work

- **`compare_runs.py`.** This sweep's real value is as the baseline a
  continuous-batching run gets diffed against, and the README's first rule
  (one variable per comparison) deserves a tool that enforces it across two
  folders the way `report.py` now enforces it within one.
- **Repeats.** Three runs on a handful of cells would put an error bar on the
  curves and settle the batch-16 anomaly.
- **An arrival-rate load generator.** The records already carry
  `"load": "offline-drain"` to keep this honest. Adding a Poisson arrival mode
  is what makes TTFT, inter-token latency and p99 mean something — and latency
  is where continuous batching's advantage is largest and this sweep is
  currently silent.
- **A sequential-mode baseline** in the same folder, so the naive cache appears
  on the same axes rather than being approximated by batch 1.

### Further out

FP8 or INT8 weights (Finding 3 says decode is bandwidth-bound, so halving
weight bytes should be close to a 2× decode win — and MBU is the metric that
would prove it), longer-context workloads where KV rather than weights
dominates the byte budget, tensor parallelism across multiple GPUs, and
speculative decoding.
