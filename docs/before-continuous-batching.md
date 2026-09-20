# Before Continuous Batching

Everything worth doing before the scheduler work starts. Continuous batching and
paged attention get their own documents; this one is about the engine underneath
them, which is currently giving away roughly two thirds of every decode step.

Replaces `static-batching-on-h100.md` and `raising-decode-mbu.md`. The baseline
sweep's tables are not copied here — they regenerate from the committed records:

```bash
python -m benchmarks.report benchmarks/results/qwen3-8b-bfloat16-n512-seed0-cuda
```

---

## 1. What the measurements say

### 1.1 The baseline sweep

Qwen3-8B, bfloat16, 1× H100 80GB, 3 workloads × 10 batch sizes, 26 trials
completed, commit `faa672214e4f`.

Four findings set the rest of this document:

- **Throughput saturates at batch 64–128.** The last doubling to 256 buys 14%,
  for ~60% more memory. The useful operating point is well below the memory
  ceiling.
- **Decode never exceeds 25% of the H100's 3.35 TB/s**, at any batch size, on
  any workload. Batching never moves it. *This is the ceiling this plan
  attacks.*
- **On `sharegpt-like`, `occ` is 0.19** — 81% of decode slots advance sequences
  that already finished, and the memory ceiling tracks the length distribution's
  tail rather than its mean. These are continuous batching's and paged
  attention's targets respectively, and **nothing in this plan touches them.**
- **Prefill is 36% of wall time at batch 256.** Amdahl is already binding on the
  best-case workload, so decode-side work has a visible ceiling.

The two ceilings are independent and multiply. A 4–5× implementation win stacks
on top of whatever the scheduler buys.

### 1.2 The decode profile

Record: [`20260823-020026-cuda-static-uniform-512x128-54225e.json`](../benchmarks/results/20260823-020026-cuda-static-uniform-512x128-54225e.json).
One decode step under `torch.profiler`, Qwen3-0.6B, **float32**, batch 64,
`kv_len` 513.

**Decode does not use FlashAttention.** It dispatches to
`aten::_scaled_dot_product_attention_math`, and the step decomposes as:

| Kernel | GPU time | Share | What it is |
|---|---|---|---|
| `elementwise` (via `repeat_interleave`) | 11.95 ms | 34% | GQA expansion: K and V physically duplicated from 8 heads to 16 |
| `vectorized_gather` (via `aten::index`) | 5.43 ms | 15% | the KV slot gather at [`preallocated.py:177`](../src/tiny_llm_serve/kv/preallocated.py) |
| `vectorized_elementwise` (via `aten::mul`) | 5.07 ms | 14% | scaling Kᵀ by 1/√d as a separate full-tensor pass |
| `gemv` (QKᵀ) | 4.06 ms | 12% | |
| `gemv` (PV) | 2.74 ms | 8% | |
| `sm80_xmma_gemm` | 2.49 ms | 7% | all 113 linear layers |
| `elementwise` | 0.88 ms | 3% | RMSNorm / RoPE / SwiGLU |

Total 35.2 ms of GPU time. **Attention alone is 70% of it, and the model's
actual matmuls are 7%.**

The kernels are not slow — each runs at 1.9–3.0 TB/s against a 3.35 TB/s peak.
The problem is byte count: the step moves **68 GB where the minimum is 7.5 GB**,
a 9× amplification. `repeat_interleave`, `aten::index` and `aten::mul` are all
pure overhead that a fused attention kernel would never incur.

### 1.3 Why the fallback happens

Three independent disqualifiers, all checkable against
`torch/include/ATen/native/transformers/sdp_utils_cpp.h` in the installed wheel:

1. **float32.** [`bench_throughput.py:367`](../benchmarks/bench_throughput.py)
   defaults `--dtype` to `float32`. Flash and cuDNN are fp16/bf16 only.
2. **The explicit `attn_mask`.** `check_for_attn_mask` (line 260) rejects flash
   for *any* non-null mask, regardless of shape or contents. FlashAttention
   exists so the `[B,H,q,kv]` score matrix never reaches HBM; an arbitrary mask
   is a tensor of that shape *in* HBM. It supports only masks computable from
   indices (causal, sliding window, ALiBi) and, for ragged batches, a length
   vector.
3. **`enable_gqa=True`.** The memory-efficient backend is compiled
   `supports_gqa=false` (line 397), which was the last candidate standing.

**bfloat16 alone does not fix this.** The mask disqualifies flash at any dtype,
so decode stays on the math path. What bf16 *does* fix is prefill, which passes
`attn_mask=None, is_causal=True` and therefore becomes flash-eligible the moment
the dtype is right.

Two caveats on transferring these numbers: the profile is 0.6B in fp32 while the
published baseline is 8B in bf16, so the *shares* carry and the absolutes do
not. And the GQA expansion is 2× on 0.6B (16/8 heads) but **4× on 8B** (32/8) —
the model actually benchmarked is worse than the one profiled.

To check a backend decision directly instead of inferring it from kernel names:

```python
from torch.backends.cuda import SDPAParams, can_use_flash_attention
can_use_flash_attention(SDPAParams(q, k, v, attn_mask, 0.0, is_causal, True), debug=True)
```

### 1.4 Scope change: custom kernels are in

Earlier docs claimed a "no custom CUDA, no FlashInfer" constraint. **Custom
kernels are now in scope.** Third-party inference engines — vLLM, TensorRT-LLM,
FlashInfer, the `flash-attn` package — stay out.

The original rationale is unchanged and, read carefully, argues *for* writing
our own: every layer should be inspectable so a measurement traces to the line
that caused it. `F.scaled_dot_product_attention` is the opposite of that — it
took a profiler run and a dive through C++ headers to learn which of four
branches it had silently taken. A Triton kernel under
`src/tiny_llm_serve/kernels/` is more inspectable, not less. Commit `ce85031`
already said real kernels land there next.

---

## 2. The plan

| # | Step | Where | Expected |
|---|---|---|---|
| 1 | bfloat16 | one flag | ~2× on bytes; prefill onto flash |
| 2 | De-sync the decode loop, freeze its shapes | `llm_engine.py`, `preallocated.py` | 10–25%; prerequisite for 5, 6, 7 |
| 3 | KV view instead of gather | `preallocated.py` | 15% of the step, for 3 lines |
| 4 | Decode attention that takes a length vector | `kv/base.py`, `attention.py`, `qwen3.py` | removes ~⅔ of the step |
| 5 | **Our own Triton decode kernel** | `src/tiny_llm_serve/kernels/` | the point of the repo |
| 6 | `torch.compile` the decoder layer | `models/qwen3.py` | 1.2–1.5× |
| 7 | CUDA graphs on the decode step | needs 2 | 2–4× at bs ≤ 64 |

Steps 1–3 are small and land quickly. Step 5 is the substantial one. Steps 6–7
are overhead removal and are independent of the attention work.

### The device seam

Steps 4 and 5 are both CUDA-only — `varlen_attn` is FlashAttention underneath,
and Triton does not run on a Mac. The engine still has to run on CUDA, MPS and
CPU from the same code, as the README promises and as CI depends on. That holds,
because the two halves of the change separate cleanly:

- **The KV representation is device-agnostic.** A pool tensor, a `block_table`
  and a `seqused_k` vector are just data. Nothing about them needs a GPU.
- **Only the consumer differs.** `Attention` becomes a small dispatcher over
  backends that all take the same inputs and return the same thing:

| Backend | Device | What it is |
|---|---|---|
| `reference` | any | gather from the pool, build the mask from `seqused_k`, call `F.scaled_dot_product_attention` |
| `varlen` | CUDA | step 4 |
| `triton` | CUDA | step 5, ours |

The `reference` backend is not new code. It is today's `_DecodeStep.append`
gather plus today's mask construction, **relocated** from the KV manager into an
attention backend, where it becomes one implementation among several rather than
the thing every device is forced through. It is pure PyTorch and runs anywhere.

Three consequences worth planning around:

- **Resolve the backend once**, at engine construction, the way
  `loader.resolve_device` already resolves the device. Not a device sniff per
  forward pass, and not a branch inside the model — [`qwen3.py`](../src/tiny_llm_serve/models/qwen3.py)
  should stay ignorant of which backend it got, exactly as it is ignorant of
  which KV backend it got.
- **Record it.** Add `attn_backend` to `run_config`
  ([`bench_throughput.py:227`](../benchmarks/bench_throughput.py)) and a
  `disagree("attn_backend", …)` check beside the `dtype` one at
  [`report.py:204`](../benchmarks/report.py). A folder mixing two attention
  backends is not one comparison, and the first rule of the harness is that it
  has to be.
- **Prefill stays on SDPA on every device.** The kernel in step 5 is a decode
  kernel — `q_len == 1` is what makes it tractable. Prefill passes no mask and
  gets flash on CUDA in bf16 already (step 1). One kernel, not two.

The payoff is larger than portability. A `reference` backend that runs on the
Mac means the kernel's *semantics* — block-table indirection, `seqused_k`
boundaries, the partially-filled last block, the GQA head mapping — are testable
locally with `@backends("all")` from [`tests/backends.py`](../tests/backends.py).
Only *performance* work needs a pod, which is what §3 is built around.

This is also how production engines are structured: vLLM selects among
FlashAttention, FlashInfer, Torch SDPA and CPU backends behind one interface.
The seam is standard, not a concession.

### Step 1 — bfloat16

Pass `--dtype bfloat16` and make it the default in
[`bench_throughput.py:367`](../benchmarks/bench_throughput.py). The checkpoint's
own `config.torch_dtype` is already `bfloat16`; the CLI default is overriding it.

Halves every byte moved, halves the KV reservation (9.4 GB → 4.7 GB at the
profiled shape, which is headroom for a larger batch), and moves prefill onto
the flash kernel. Decode stays on the math path until step 4.

**Verify:** `kv_bytes_reserved` halves; prefill time drops; the profile's
`sm80_xmma_gemm_f32f32...` kernel is replaced by a bf16 one.

### Step 2 — de-sync the decode loop and freeze its shapes

Every site below reads device state on the host, draining the pipeline and
stopping the host from running ahead:

| Site | Cost |
|---|---|
| [`llm_engine.py:206`](../src/tiny_llm_serve/engine/llm_engine.py) — `if finished[i]` inside a per-row Python loop | **one sync per sequence per step** |
| [`llm_engine.py:205`](../src/tiny_llm_serve/engine/llm_engine.py) — `next_ids.tolist()` | one sync |
| [`llm_engine.py:215`](../src/tiny_llm_serve/engine/llm_engine.py) — `bool(finished.all())` | one sync |
| [`preallocated.py:84,86`](../src/tiny_llm_serve/kv/preallocated.py) — two `.any()` guards | two syncs |
| [`preallocated.py:152`](../src/tiny_llm_serve/kv/preallocated.py) — `int(write_pos.max())` | one sync |

There is a sixth problem that is not a sync but matters more for what follows:
[`llm_engine.py:220`](../src/tiny_llm_serve/engine/llm_engine.py) grows
`seen_ids` with `torch.cat` every step. With `repetition_penalty == 1.0` — the
entire benchmark path — `seen_ids` is never read: the sampler guards on the
penalty, and the loop only wants `seen_ids[:, 0]`, a fixed `[batch]` tensor. It
is a per-step allocate-and-copy of a tensor **whose shape changes every step**,
for nothing.

Changing shapes are what block graph capture and force endless recompiles, so
fix both together:

- Keep a `[batch]` first-token tensor for the idle feed; preallocate a
  `[batch, max_len]` buffer written in place for when the penalty *is* on.
- Take `_kv_len` from host state (`max(prompt_lens) + step`) — the host already
  knows it; no device read is needed.
- Round `kv_len` up to a bucket (multiples of 128 or 256) so decode sees a
  handful of shapes instead of one per step.
- Keep `finished` on device, materialize outputs at the end. Under `ignore_eos`
  the finish condition is a step count the host already knows.
- Move the two `.any()` validation guards out of the hot path.

**Verify:** the microbenchmark's dispatch-only time and synchronized time stop
tracking each other at bs=1.

### Step 3 — KV view instead of gather

`_DecodeStep.append` ends by returning
`k_cache[self._slot_ids, : self._kv_len]`. `slot_ids` is a tensor, so this is
**advanced indexing — a copy, not a view.** Verified:

```
advanced-index is a copy: True
basic-slice is a view:   True
slice is contiguous:     False
```

Each layer therefore reads the KV window, writes a full duplicate, and attention
reads the duplicate — 3× the minimum KV bytes, twice per layer. That is the
5.43 ms `vectorized_gather` row in §1.2, 15% of the step.

`admit` pops from the end of `_free_slots` ([`preallocated.py:64`](../src/tiny_llm_serve/kv/preallocated.py)),
so slots come out as `[n-1, …, 0]` — permuted for no reason. Hand them out in
order (`_free_slots = list(reversed(range(num_slots)))`) and, **in static mode
where the batch is the whole pool**, `append` can return `k_cache[:n, :kv_len]`,
a genuine view.

Step 4 makes this moot on the CUDA path, so it is fair to ask why it is a step
at all. Three reasons: it is three lines against a measured 15%; it is the one
attention-related win that does not depend on `varlen_attn` behaving as
documented, which is still unverified; and it carries forward into the
`reference` backend, which keeps gathering from the pool on every device step 4
does not cover. Do it first — it is insurance, cheaply bought.

It does not survive continuous batching: once slots are freed and reused out of
order the batch stops being a contiguous prefix of the pool. That is an argument
for the block table in steps 4–5 being the durable answer, not against doing
this now.

**Verify:** no `contiguous` kernel appears in the trace. The slice is strided,
and a backend that materializes it has moved the copy rather than removed it.

### Step 4 — decode attention that takes a length vector

The mask is the problem, and the fix is to stop passing one. Fused kernels
express ragged lengths as a **`[batch]` length vector**, not a
`[batch,1,1,kv_len]` mask — that is what flash, paged kernels and our own kernel
in step 4 all want.

torch 2.13 ships this as a first-party op, `torch.nn.attention.varlen.varlen_attn`,
which takes `seqused_k` ("number of valid KV tokens per batch element … useful
for KV-cache decoding where the cache slot is larger than the actual sequence"),
`block_table`, `enable_gqa` and `num_splits`, and dispatches to
`aten::_flash_attention_forward`.

**`PreallocatedKVManager` is already a valid page pool for it.** The cache is
`[num_slots, max_model_len, num_kv_heads, head_dim]`; set `page_size =
max_model_len` and `total_pages = num_slots` and that is exactly the pool shape,
with one page per sequence:

```python
# UNVERIFIED — confirm on the pod before building on it, in particular
# whether cu_seq_k should be None on the paged path.
cu_seq_q    = torch.arange(B + 1, dtype=torch.int32, device=dev)  # one token per seq
block_table = slot_ids.to(torch.int32).unsqueeze(1)               # [B, 1]
seqused_k   = cached_seq_lens[slot_ids].to(torch.int32)           # [B]
out = varlen_attn(q, k_cache[l], v_cache[l], cu_seq_q, None, 1, max_k,
                  enable_gqa=True, seqused_k=seqused_k, block_table=block_table)
```

This removes the mask, the GQA expansion, the separate scale pass **and the slot
gather** — the kernel reads the cache in place through the block table, so
permuted slots cost nothing and step 3's view becomes unnecessary on this path.

**This is a `KVCacheView` protocol change, not an `attention.py` edit.** Today
[`base.py:26`](../src/tiny_llm_serve/kv/base.py) requires `append` to *return*
the k/v to read over, and [`qwen3.py:73`](../src/tiny_llm_serve/models/qwen3.py)
consumes them as `k, v = kv_cache.append(...)`. That return is what forces the
materialization. A view must instead expose the pool plus its metadata —
`block_table`, `seqused_k` — and `append` becomes write-only. Budget for four
files: `base.py`, `preallocated.py`, `naive.py`, `qwen3.py`. The naive cache can
keep returning dense tensors behind the same protocol; it backs sequential mode
and is not on this path.

It also means **paged attention later is a `page_size` change, not a kernel
change.** Shrink pages to 16 and widen the block table; the call site is
identical. That keeps the paging milestone where it belongs — in the allocator
and the scheduler.

Constraints: CUDA + fp16/bf16 only, and `seqused_k`/`block_table` are
inference-only — so this lands as the `varlen` backend behind the seam above,
with `reference` carrying CPU and MPS.

Build the seam and the `reference` backend *before* wiring `varlen` in. That
ordering means the protocol change and the new attention implementation are two
separately-testable commits rather than one change that is red on every device
at once, and `reference` is the allclose target for `varlen` the moment it
arrives.

**Verify:** allclose against the current SDPA path on ragged lengths, then
confirm `repeat_interleave`, `aten::index` and `aten::mul` are gone from the
trace.

### Step 5 — our own Triton decode kernel

The reason step 4 comes first is that it leaves a production-grade reference
running in the same process. A kernel written with nothing to compare against
cannot be told from a 4× miss.

Decode attention is the right first real kernel: `q_len == 1`, so there is no
intra-tile causal logic and the whole thing is a reduction over `kv_len`. GQA is
one line (`kv_head = q_head // group`). The length vector is a `tl.load` and a
comparison.

**Take the same interface as step 4** — cache pool, `block_table`, `seqused_k` —
so it drops into the same call site and inherits the same paging story.

The part actually worth learning is **split-K (flash-decoding)**: at low batch,
`batch × heads` programs do not fill an H100, so the kernel must parallelize
over the KV dimension and combine partial softmax results via log-sum-exp. That
is the mechanism behind the flat 23.5 ms/step from batch 1 to 64 in the
baseline, and `varlen_attn`'s `num_splits` is the same idea — which makes it a
directly comparable reference.

**The scoreboard already exists.** `decode_bytes_read` is the roofline, and the
kernel's own floor is reading K and V exactly once. Report GB/s against 3.35
TB/s the way `kernels/scratch/` already does with `triton.testing.do_bench`;
70–85% of peak is the target band for a good decode kernel.

**Verify, in two places.** Semantics against `reference` — ragged lengths, a
`kv_len` that is not a multiple of the block size, a batch whose slots are
permuted, a `group` of both 2 and 4. Those are the bugs that cost the most time
and none of them need a GPU to find; `reference` runs on the Mac, so write these
first and iterate locally. Then performance against `varlen` on the pod, which
is the only part that needs renting anything.

### Step 6 — `torch.compile` the decoder layer

Roughly 55–60 aten calls per layer — RMSNorm is ~7 (the fp32 upcast at
[`layernorm.py:16`](../src/tiny_llm_serve/layers/layernorm.py)), rotary's
chunk/mul/cat chain is ~20. Across 28–36 layers that is on the order of 2,000
dispatches per step, which is most of why bs=1 is ~5× off the bandwidth ideal.

Needs step 2 first: a host sync is a graph break, and each new `kv_len` is a
recompile without the shape buckets.

### Step 7 — CUDA graphs

Try `torch.compile(mode="reduce-overhead")` first as a cheap upper bound on what
capture is worth. Hand-rolling it — one graph per batch-size bucket, the way
vLLM does — is the better writeup.

The old plan deferred this until after paged attention, on the grounds that the
buffers and index math get redone when the KV layout changes. **Step 4 removes
that objection:** with paging as a `page_size` change, the layout does not move,
so graphs are safe to capture before the scheduler work.

---

## 3. Measuring

### The decode-step microbenchmark

Every step above moves one number: `s_per_decode_step` at a given
`(batch, kv_len)`. Draining a whole workload to read it is waste — the sweep
spends 88% of its time on batch sizes that answer nothing, and
`sharegpt-like` at bs 1/2/4 alone is 1.6 hours.

One prerequisite that pays for itself three times: **extract a single decode
step out of `generate_batch_ids` into a callable** (`step(state) -> next_ids`).
The microbenchmark needs it, graph capture needs it, and the continuous-batching
scheduler needs it anyway.

```bash
python -m benchmarks.bench_decode_step --model Qwen/Qwen3-8B --dtype bfloat16 \
    --cells 1x576,8x576,64x576,256x576,64x2048 --warmup 10 --steps 30 --repeats 3
```

- One real prefill to build the state, then W warmup steps, then N timed steps.
  Warmup stops being optional once `torch.compile` is involved.
- Time the **engine** step, not the model forward: the syncs live in the loop.
- Per cell report median ms/step, **dispatch-only ms/step**, effective GB/s and
  MBU. The host/device split is what attributes a regression to a cause.
- `kv_len=576` is the mean of `uniform-512x128` over its run, so numbers land
  next to the baseline. Both large cells fit an 80 GB card.
- Fill the cache with random values rather than leaving it zeroed, and write
  records under `benchmarks/results/decode-step/`.

### Milestone runs

A trimmed sweep at each milestone, not after each step:

```bash
python -m benchmarks.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
    --num-requests 128 --batch-sizes 8,32,64,128 \
    --run-name qwen3-8b-bf16-n128-seed0-cuda-<change-name>
```

`--run-name` is mandatory: `recorded_trials()` keys a trial on
model/dtype/num_requests/seed/workload/mode/batch_size and **not** on the commit,
so re-running the same grid after an optimization would skip every cell as
already recorded. One folder per code version. The committed n=512 sweep stays
the published baseline; re-run the full ladder exactly once, on the final code.

Keep the HF cache on a network volume (`HF_HOME`) — downloading 16.4 GB at pod
start is billed GPU time, and it becomes the largest line item once runs are
minutes. Develop against Qwen3-0.6B locally, then rent once per batch of
changes.

### The correctness gate

Steps 2, 3, 4, 5 and 7 are all silent-corruption risks — a view aliasing a stale
write, a captured graph reading a moved buffer, a kernel with an off-by-one on
the last partial block. The gate already exists:
`test_static_batch_matches_sequential_on_qwen`
([`tests/test_llm_engine.py`](../tests/test_llm_engine.py)) runs ragged prompts
on CPU or MPS. Extend it to cover the compiled and captured paths, and run it
before every rental. It is the difference between "1.5× faster" and "1.5× faster
and wrong."

Add one test the seam makes possible: **backend parity** — the same inputs
through every backend available on this machine, asserted allclose. On CI and on
the Mac that exercises `reference` alone and is nearly free; on a CUDA box the
same test file covers `varlen` and `triton` with no extra code, because
`@backends("all")` already drops what is not available. That is the gate that
catches a kernel which is fast and subtly wrong, and it is the reason to build
the seam in step 4 rather than bolting it on in step 5.

---

## 4. Not in this plan

**Deferred scheduler work**, because both would eat continuous batching's
headline result: length-sorted waves (three lines in `waves()`, and it collapses
the reservation toward the mean), and compacting finished rows out of a wave
(~20 lines, drives `occ` from 0.19 toward 1.0 without admitting anything new).
The second is worth keeping as the ablation *inside* the continuous-batching
document — "drop finished" against "drop finished and admit new" is the more
interesting split.

**Out of scope until after the scheduler work:** chunked prefill (it belongs
with continuous batching, and Finding 7 says prefill is the next ceiling),
FP8/INT8 weights, tensor parallelism, speculative decoding.

**A framing note to carry forward:** FP8/INT8 will *lower* reported MBU —
quantization moves fewer bytes for the same work, so the ratio falls while
throughput rises. MBU is a diagnostic for overhead, not a score to maximize. And
the realistic end state for this plan is ~60% MBU, not 90%; production engines
reach 60–80% on memory-bound decode.
