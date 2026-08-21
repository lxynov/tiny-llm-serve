# Raising Decode MBU

A code-level follow-up to
[`static-batching-on-h100.md`](static-batching-on-h100.md), which measured
decode at **18–25% of the H100's 3.35 TB/s at every batch size and on every
workload** and attributed the gap to "eager PyTorch" without saying which part
of it. This document names the parts.

Short version: **"MBU 25%" is two different diseases with the same symptom**,
and they dominate at opposite ends of the batch axis. One of them is visible in
the source.

---

## 1. The metric hides an amplification factor

`decode_bytes_read` ([`benchmarks/bench_throughput.py:130`](../benchmarks/bench_throughput.py))
is the *minimum* bytes decode must move, derived from weight and KV sizes.
MBU is that modelled minimum ÷ elapsed time. So a low MBU means one of two
things, and they are not the same problem:

- **the machine is idle** — the bytes are not being moved fast enough; or
- **the machine is busy moving bytes the model does not count.**

Both are happening here:

| Regime | Disease |
|---|---|
| bs = 1 | The GPU is genuinely idle. Host dispatch is the pacer. |
| bs ≥ 64 | The GPU is busy, but roughly half its KV traffic is a copy that does not need to exist. |

The profiler distinguishes them, and the fixes are unrelated.

---

## 2. Decode copies the whole KV window, every layer, every step

[`kv/preallocated.py:175`](../src/tiny_llm_serve/kv/preallocated.py) —
`_DecodeStep.append` ends with:

```python
return (k_cache[self._slot_ids, : self._kv_len],
        v_cache[self._slot_ids, : self._kv_len])
```

`slot_ids` is a tensor, so this is **advanced indexing — a copy, not a view**.
Confirmed locally:

```
advanced-index is a copy: True
basic-slice is a view:   True
```

Each layer therefore reads the KV window, writes a full duplicate of it, and
SDPA then reads the duplicate: **3× the minimum KV bytes**, 72 times per step
(k and v, 36 layers).

On `uniform-512x128` (mean kv_len 576 over the run, 144 KiB of KV per token,
16.4 GB of weights):

| bs | charged bytes/step | actual bytes/step | reported MBU | real MBU |
|---|---|---|---|---|
| 64 | 21.8 GB | 32.7 GB | 0.225 | ~0.34 |
| 256 | 38.1 GB | 81.6 GB | 0.182 | ~0.39 |

At bs=256 more than half the traffic is the redundant copy. This is also, most
likely, much of what Finding 2 of the baseline doc reads as "the step has
become compute- and KV-bound" between batch 128 and 256 — a good part of that
knee is the copy scaling with the batch.

### The fix is small

`admit` ([`preallocated.py:60`](../src/tiny_llm_serve/kv/preallocated.py)) pops
from the end of `_free_slots`, so slots come out as `[n-1, …, 0]` — a
permutation for no reason. Hand them out in order
(`_free_slots = list(reversed(range(num_slots)))`), and in static mode the
batch *is* the whole pool, so `append` can return `k_cache[:n, :kv_len]` — a
genuine view, with SDPA reading the cache in place.

One thing to verify afterwards: the view is strided and non-contiguous, and if
the chosen SDPA backend materializes it, the copy has been moved rather than
removed. Watch the profiler for a `contiguous` kernel.

---

## 3. The decode loop syncs ~batch + 5 times per step

Every one of these reads device state on the host, which drains the pipeline
and stops the host from running ahead of the GPU:

| Site | Cost |
|---|---|
| [`llm_engine.py:177`](../src/tiny_llm_serve/engine/llm_engine.py) — `if finished[i]` inside a per-row Python loop | `finished` is a device tensor: **one sync per sequence per step** — 256 of them at bs=256 |
| `llm_engine.py:176` — `next_ids.tolist()` | one sync |
| `llm_engine.py:186` — `bool(finished.all())` | one sync |
| [`preallocated.py:84,86`](../src/tiny_llm_serve/kv/preallocated.py) — two `.any()` guards | two syncs |
| `preallocated.py:152` — `int(write_pos.max())` | one sync |

`_kv_len` is the easy one: the host already knows it (`max(prompt_lens) +
step`), so no device read is needed at all. Finish-detection should stay
on-device with outputs materialized at the end, or sync every *k* steps.

Worth maybe 10–25% on its own. Its real value is that it is the
**prerequisite for CUDA graphs** — a step that reads device state on the host
cannot be captured.

---

## 4. Why bs=1 is 4.8× off the bandwidth ideal

Counting aten calls in the decode path: roughly 55–60 per layer. RMSNorm alone
is ~7 (the fp32 upcast at
[`layers/layernorm.py:16`](../src/tiny_llm_serve/layers/layernorm.py)), and
rotary's chunk / mul / cat chain is ~20. Across 36 layers that is **~2,000
dispatches per step**. At the usual 8–12 µs of Python + dispatcher + launch per
op, that is ~20 ms against a measured 23.5 ms. The bandwidth-ideal step is
4.9 ms.

This sharpens Finding 2 of the baseline doc. `ms/step` is flat from bs=1 to
bs=64 **not because the GPU has bandwidth to spare, but because the host is the
pacer and the GPU is idle underneath it.** The batch "riding along free" is the
GPU filling in time the host was going to spend anyway.

One measurement settles it: time the step with no sync (pure dispatch time),
then synchronize and time again. If the two match, decode is host-bound.

```python
t = time.perf_counter(); model(...); cpu = time.perf_counter() - t   # dispatch only
torch.cuda.synchronize();            wall = time.perf_counter() - t   # + device
```

---

## 5. Priority

| | Change | Where | Expected |
|---|---|---|---|
| 1 | Profile one decode step; check which SDPA backend decode picks | — | decides everything below |
| 2 | KV view instead of gather | `preallocated.py:175` | ~1.5× at bs=256 |
| 3 | De-sync the decode loop | `llm_engine.py:176-186` | 10–25%, unblocks 4 |
| 4 | CUDA graphs on the decode step | needs 3 + static buffers | 2–4× at bs≤64 |
| 5 | `torch.compile` the decoder layer | `models/qwen3.py` | 1.2–1.5× on top |

Items 2 and 3 are small and independent of each other; 2 is measurable against
the existing sweep with one variable changed.

### Check first: which SDPA backend does decode use?

Decode passes an explicit `attn_mask`
([`kv/preallocated.py:154`](../src/tiny_llm_serve/kv/preallocated.py)), which
**disqualifies the FlashAttention backend** in
`F.scaled_dot_product_attention`. If SDPA is falling back to the math path,
`enable_gqa=True` there materializes k/v at 32 heads instead of 8 — another 4×
on the KV term, which would be the whole story at high batch. Five minutes with
a profiler trace (or `torch.nn.attention.sdpa_kernel`) settles it, and it
should come before any of the work above.

---

## 6. Two framing notes

- **FP8/INT8 weights will lower reported MBU.** Quantization roughly doubles
  decode throughput but moves fewer bytes for the same work, so the ratio
  falls. MBU is a diagnostic for *overhead*, not a score to maximize —
  throughput is the thing being bought.
- **The realistic target is ~60%, not ~90%.** Production engines reach 60–80%
  MBU on memory-bound decode. 25% → 60% is what the items above are worth; the
  rest needs kernels this repo has deliberately excluded.
