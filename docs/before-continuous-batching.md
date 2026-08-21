# Before Continuous Batching

A work plan derived from [`static-batching-on-h100.md`](static-batching-on-h100.md)
and [`raising-decode-mbu.md`](raising-decode-mbu.md). Those two documents
measured the engine and named what is wrong with it; this one answers the two
questions that come next. **What is worth implementing before continuous
batching and paged attention**, and **what to measure after each change without
renting an H100 for four and a half hours every time.**

Short version: three implementation items come before any scheduling change,
because they are what Finding 3 of the baseline doc is made of and they
multiply with everything after them. And the measurement moves off the sweep
onto a decode-step microbenchmark — the sweep spends **88% of its time on batch
sizes that answer nothing**, and the number every item below moves is one the
microbenchmark reads in seconds.

---

## 1. Order of work

### 0. Profile first — it re-orders everything below

Twenty minutes on one pod, before writing any optimization code. Three
measurements:

- **Which SDPA backend does decode pick?** [`layers/attention.py:46`](../src/tiny_llm_serve/layers/attention.py)
  passes the explicit `attn_mask` built in
  [`kv/preallocated.py:154`](../src/tiny_llm_serve/kv/preallocated.py), which
  disqualifies FlashAttention. If decode is falling back to the *math* path,
  `enable_gqa=True` materializes k/v at 32 heads instead of 8 — the KV term is
  then 4× the modelled minimum, which would be the whole story at bs ≥ 64 and
  would promote item C to the top of the list.
- **The host-bound test at bs=1.** Dispatch-only time against synchronized
  time (§4 of the MBU doc). If they match, decode is paced by the host and
  item D is the largest number available.
- **One `torch.profiler` trace of a decode step.** Gaps between kernels are
  launch overhead; a `contiguous` or `copy` kernel per layer is the gather at
  [`preallocated.py:175`](../src/tiny_llm_serve/kv/preallocated.py).

### The four items

| | Change | Where | Worth | Why this position |
|---|---|---|---|---|
| **A** | De-sync the decode loop, and stop its shapes changing every step | [`llm_engine.py:170-190`](../src/tiny_llm_serve/engine/llm_engine.py), [`preallocated.py:84,86,152`](../src/tiny_llm_serve/kv/preallocated.py) | 10–25% | Prerequisite for C and D. Nothing else lands until the step stops reading device state |
| **B** | KV view instead of gather | [`preallocated.py:175`](../src/tiny_llm_serve/kv/preallocated.py) | ~1.5× at bs=256 | Smallest diff in the list, largest immediate number |
| **C** | Fuse the layer: `torch.compile`, and evaluate `flex_attention` | [`models/qwen3.py`](../src/tiny_llm_serve/models/qwen3.py), [`layers/attention.py`](../src/tiny_llm_serve/layers/attention.py) | 1.2–1.5×, more if decode is on the math path | Survives the paged-attention rewrite; hand-rolled graph capture will not |
| **D** | CUDA graphs on the decode step | needs A + B | 2–4× at bs ≤ 64 | Biggest single number, most rework risk |

### A — de-sync, and freeze the shapes

The MBU doc lists five sync sites. There is a sixth it misses, and it matters
more than any of them for what comes after:
[`llm_engine.py:183`](../src/tiny_llm_serve/engine/llm_engine.py) grows
`seen_ids` with `torch.cat` every step. With `repetition_penalty == 1.0` — the
entire benchmark path — `seen_ids` is never read: the sampler guards on the
penalty at [`layers/sampler.py:37`](../src/tiny_llm_serve/layers/sampler.py),
and the loop itself only wants `seen_ids[:, 0]` for the idle feed, which is a
fixed `[batch]` tensor. So it is a per-step allocate-and-copy of a tensor
**whose shape changes every step**, for nothing.

That, plus the mask's growing `kv_len`, is what will block graph capture and
force endless `torch.compile` recompiles. Fix both together:

- Keep a `[batch]` first-token tensor for the idle feed, and a preallocated
  `[batch, max_len]` buffer written in place for when the penalty *is* on.
- Take `_kv_len` from host state (`max(prompt_lens) + step`) rather than
  `int(write_pos.max())` — the host already knows it.
- Round `kv_len` up to a bucket (multiples of 128 or 256) so decode has a
  handful of distinct shapes instead of one per step. The mask already hides
  the slack.
- Keep `finished` on device and materialize outputs at the end, or every *k*
  steps. Under `ignore_eos` the finish condition is a step count the host
  knows, so the whole per-row `finished[i]` loop can go.
- Move the two `.any()` validation guards out of the hot path.

### B — KV view instead of gather

`admit` ([`preallocated.py:60`](../src/tiny_llm_serve/kv/preallocated.py)) pops
from the end of `_free_slots`, so slots come out permuted for no reason. Hand
them out in order and, in static mode where the batch *is* the whole pool,
`append` can return `k_cache[:n, :kv_len]` — a basic slice, a genuine view, with
SDPA reading the cache in place instead of a full duplicate per layer per step.

Verify afterwards that no `contiguous` kernel appeared in the trace: the view is
strided, and a backend that materializes it has moved the copy rather than
removed it.

### C — fuse the layer

`torch.compile` on the decoder layer, to collapse the RMSNorm / RoPE / SwiGLU
chain that currently runs as separate kernels. It needs A first (a host sync is
a graph break) and the shape buckets from A (each new `kv_len` is a
recompile).

Evaluate [`torch.nn.attention.flex_attention`](https://pytorch.org/blog/flexattention/)
here too. It is the principled answer to "an explicit `attn_mask` disqualifies
the Flash backend": it compiles a per-row valid-length mask into a fused
Flash-style kernel with GQA support, and it ships with torch — so it stays
inside this repo's no-custom-CUDA, no-FlashInfer constraint. If step 0 finds
decode on the math path, this is the single biggest item in the document.

### D — CUDA graphs

`torch.compile(mode="reduce-overhead")` gives CUDA graphs almost free; run it
first as a cheap upper bound on what capture is worth. Hand-rolling it — one
graph per batch-size bucket, the way vLLM does — is the better writeup, but the
buffers and index math get redone when paged attention changes the KV layout.

**The call: A, B and C before continuous batching; hand-rolled graphs after
paged attention**, unless D is wanted as its own document and the rework is
acceptable.

### Deliberately not yet

Two cheap wins that should wait:

- **Length-sorted waves.** Three lines in `waves()`. Because `max_model_len` is
  `max(prompt) + max(output)` per wave, sorting collapses the reservation
  toward the mean and attacks Finding 6's memory ceiling directly.
- **Compacting finished rows out of the wave.** ~20 lines, and it drives `occ`
  from 0.19 toward 1.0 without admitting anything new.

Both are scheduler changes, and both would eat continuous batching's headline
result. The second is worth keeping as the ablation *inside* the
continuous-batching document: "drop finished" against "drop finished and admit
new" is the more interesting split.

Also out of scope until after the scheduling work: FP8/INT8 weights, chunked
prefill (it belongs with continuous batching), tensor parallelism, speculative
decoding.

---

## 2. Measuring each step

### The sweep's cost is not where it looks

From this repo's own [`sweep.json`](../benchmarks/results/qwen3-8b-bfloat16-n512-seed0-cuda/sweep.json):

| Subset | Trials | Pod time | Share of the sweep |
|---|---|---|---|
| Full grid | 26 | **4.5 h** | 100% |
| bs ≥ 32 only | 15 | 32 min | 12% |
| bs ≥ 64 only | 12 | 20 min | 7% |

`sharegpt-like` at bs=1, 2 and 4 alone is 1.6 hours — three cells that drain 512
requests one at a time to re-derive a number already known, and one the
microbenchmark below reads in seconds. **Batch sizes ≤ 8 should not go through
the sweep again until the final published run.**

### Tier 1 — a decode-step microbenchmark, every change (~2 min, ~$0.10)

Every item in §1 moves exactly one number: `s_per_decode_step` at a given
`(batch, kv_len)`. Draining a workload to see it is waste.

One prerequisite, which pays for itself three times over: **extract a single
decode step out of `generate_batch_ids` into a callable** (`step(state) ->
next_ids`). The microbenchmark needs it, graph capture needs it, and continuous
batching's scheduler loop needs it anyway.

```
python -m benchmarks.bench_decode_step --model Qwen/Qwen3-8B --dtype bfloat16 \
    --cells 1x576,8x576,64x576,256x576,64x2048 --warmup 10 --steps 30 --repeats 3
```

- One real prefill to set the state up, then W warmup steps, then N timed
  steps. Warmup stops being optional the moment `torch.compile` is involved —
  the first steps are compilation.
- Time the **engine** step, not the model forward: the syncs and the `cat` live
  in the loop, not the model.
- Per cell, report median ms/step, **dispatch-only ms/step**, effective GB/s and
  MBU. The host/device split is what attributes a regression to a cause.
- `kv_len=576` is the mean of `uniform-512x128` over its run, so the numbers
  land next to the baseline doc's table. Both large cells fit an 80 GB card:
  256×576 is 21.7 GB of KV plus 16.4 GB of weights; 64×2048 is 19.3 GB.
- One process for the whole grid. Fresh-process-per-trial exists in
  [`sweep.py`](../benchmarks/sweep.py) for peak-memory honesty; this benchmark
  makes no peak-memory claim, and a model load per cell would *be* the cost.
- Fill the cache once with random values rather than leaving it zeroed, and
  write records under `benchmarks/results/decode-step/` so rule 4 still holds.

### Tier 2 — a trimmed sweep at milestones (~18 min, ~$0.75)

```
python -m benchmarks.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
    --num-requests 128 --batch-sizes 8,32,64,128 \
    --run-name qwen3-8b-bf16-n128-seed0-cuda-<change-name>
```

**`--run-name` is mandatory here.** `recorded_trials()` keys a trial on
model / dtype / num_requests / seed / workload / mode / batch_size — *not* on
the commit — so re-running the same grid after an optimization would skip every
cell as "already recorded". `report.py`'s `FIXED` includes `commit` and would
refuse the mixed folder anyway. One folder per code version.

Run this grid once on today's commit to establish the cheap baseline, then once
per milestone. The committed n=512 sweep stays the *published* baseline; the
full ladder gets re-run exactly once, on the final code, for the document
(4.5 h, ~$11). `benchmarks/compare_runs.py` exists on the `new-day89` branch and
is worth resurrecting to diff two of these folders.

### Tier 3 — pod hygiene, which dominates the bill once runs are minutes

- **HF cache on a network volume** (`HF_HOME`). Downloading 16.4 GB of
  Qwen3-8B at pod start is billed GPU time, and at a couple of minutes a pod it
  soon becomes the largest line item.
- **Batch the work.** Develop and debug on the Mac with Qwen3-0.6B until the
  tests pass, accumulate two or three optimizations, then rent once and measure
  them together.
- **Script the session** — microbenchmark, mini-sweep, commit the records,
  `runpodctl stop pod $RUNPOD_POD_ID` — inside tmux, so a dropped connection
  does not kill a 30-minute sweep. A pod idling while output is read costs more
  than the run did.
- **Consider an A100 80GB for the iteration loop.** Roughly 60% of the price,
  the same shapes, and every item in §1 is a relative win that shows up
  identically. [`hardware.py`](../benchmarks/hardware.py) already knows its
  2.039 TB/s, so MBU stays honest. Keep the H100 for published numbers.

### The correctness gate, which costs nothing

A, B and D are all silent-corruption risks — a view aliasing a stale write, a
captured graph reading a buffer that moved. The gate already exists:
`test_static_batch_matches_sequential_on_qwen`
([`tests/test_llm_engine.py:222`](../tests/test_llm_engine.py)) runs ragged
prompts on CPU or MPS. Extend it to cover the compiled and graph-captured paths,
and run it locally before every rental. It is the difference between "1.5×
faster" and "1.5× faster and wrong."
