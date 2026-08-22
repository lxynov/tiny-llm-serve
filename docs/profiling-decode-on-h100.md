# Profiling Decode on an H100

The measurement [`raising-decode-mbu.md`](raising-decode-mbu.md) §5 asks for
first, and [`decode-sdpa-backend.md`](decode-sdpa-backend.md) predicted from the
source. Run with
[`benchmarks/profile_decode.py`](../benchmarks/profile_decode.py) on
`Qwen/Qwen3-8B` in bfloat16, workload `uniform-512x128`, kv_len ≈ 525,
torch 2.13.0+cu130, one H100 80GB HBM3.

**Short version: decode runs on cuDNN, not math.** The static reading got the
two refusals right and the winner wrong. The consequences run the other way
from what the priority list assumed: the fused-kernel worry is closed, the KV
gather is the largest device-side waste (40% of decode's device time at
bs=256), and the host is the pacer up to bs≈64 by a wider margin than
estimated.

> **Read the absolute times with care.** This box is ~2.4× slower on the host
> than the one that produced the recorded sweep — see [§5](#5-this-box-is-not-the-sweeps-box).
> Compositions and ratios below are sound; ms/step and MBU are not comparable
> to `benchmarks/results/`.

---

## 1. The backend is cuDNN

Every decode step, at every batch size tried (1, 4, 64, 256), traces to:

```
aten::_scaled_dot_product_cudnn_attention    36.0/step
aten::_cudnn_attention_forward
```

36 calls a step, one per layer. No `aten::_scaled_dot_product_attention_math`,
and — the tell the doc asked for — **no `aten::repeat_interleave`**: cuDNN
takes the 4:1 head ratio natively, so k and v are never expanded to 32 heads.

Asked directly, with decode's exact shapes, the three fused backends answer:

| Backend | Verdict | Reason it gave |
|---|---|---|
| flash | REFUSED | *"Flash Attention does not support non-null attn_mask."* |
| mem-efficient | REFUSED | *"For dense input, both fused kernels require query, key and value to have the same num_heads. Query.sizes(): [4, 32, 1, 128], Key sizes(): [4, 8, 518, 128]"* |
| cuDNN | **usable** | — |

So `decode-sdpa-backend.md` §1–§2 hold exactly as written, up to the last row
of the table. What it got wrong is what happens next.

### Why the priority order misled

`torch._C._get_sdp_priority_order()` reports
`[FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH, CUDNN_ATTENTION, OVERRIDEABLE]`,
matching `ATen/Context.h:481-486`. Math sits ahead of cuDNN, math is always
eligible, and the conclusion followed — wrongly. **cuDNN runs anyway.** The
reported list is not the order the CUDA selector uses by default; the build
carries a `TORCH_CUDNN_SDPA_DEPRIORITIZED` knob, and setting it flips the
selection back:

```
$ python -c "...F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)..."
aten::_scaled_dot_product_cudnn_attention 1

$ TORCH_CUDNN_SDPA_DEPRIORITIZED=1 python -c "...same call..."
aten::_scaled_dot_product_attention_math 1
```

A knob that exists to *de*prioritize cuDNN is only needed if something
prioritizes it, and the trace says something does.

---

## 2. What math would have cost

Forcing the backend the source reading predicted, at bs=64, prices the
counterfactual:

| Per decode step, bs=64 | cuDNN | math (forced) |
|---|---|---|
| self device time | **16.81 ms** | **86.73 ms** (5.2×) |
| `aten::copy_` | 0.81 ms | **40.68 ms** |
| `aten::repeat_interleave` | — | 72 calls |
| `aten::clone` | — | 72 calls |
| aten calls | 8,337 | 10,893 |
| wall | 57.00 ms | 87.20 ms |
| host share | 99% | 66% |

The GQA expansion is real and it is expensive — 72 `repeat_interleave` a step
(k and v × 36 layers) turning into 40 ms of `copy_`. `decode-sdpa-backend.md`
§4 argued from the sweep's own step time that an 11× KV amplification could not
be happening. It was right, and now for a plainer reason than L2 residency:
**that path never ran.**

The practical reading: this is a *hazard*, not a diagnosis. Any change that
pushes decode off cuDNN — deprioritizing it, a shape it refuses, a torch
upgrade that reorders — lands on math and costs 5× the device time. Worth an
assertion in the engine rather than a comment.

---

## 3. The KV gather is the real copy (§2, confirmed)

`aten::index` — the advanced-indexing read at
[`preallocated.py:177`](../src/tiny_llm_serve/kv/preallocated.py) — 145 calls a
step, scaling with batch and kv_len exactly as a KV-window copy must:

| bs | device ms/step | `aten::index` | share of device time |
|---|---|---|---|
| 1 | 10.17 | 0.49 ms | 5% |
| 4 | 10.79 | 0.40 ms | 4% |
| 64 | 16.81 | 3.55 ms | 21% |
| 256 | 35.04 | **13.99 ms** | **40%** |

At bs=256 two fifths of decode's device time is a copy that
[`raising-decode-mbu.md` §2](raising-decode-mbu.md) shows how to delete. That
makes its item 2 the top device-side item — and its caveat the thing to watch:
the replacement view is strided and non-contiguous, and cuDNN may materialize
it, which would move the copy rather than remove it. The trace tells: watch
`aten::contiguous` and `aten::copy_` after the change.

---

## 4. The host is the pacer (§4, confirmed and extended)

Each step timed twice — once to the end of dispatch, once through the device:

| bs | dispatch only | through device | host share | device work |
|---|---|---|---|---|
| 1 | 56.12 ms | 56.56 ms | 99% | 10.17 ms |
| 4 | 56.61 ms | 57.05 ms | 99% | 10.79 ms |
| 64 | 56.48 ms | 57.00 ms | 99% | 16.81 ms |
| 256 | 58.41 ms | 73.11 ms | 80% | 35.04 ms |

Flat host time, growing device time, and the two only start to separate at
bs=256. The doc's framing — *"the batch rides along free because the host was
going to spend that time anyway"* — is exactly what this shows, and it holds
further up the batch axis than the doc claimed (it put the GPU as busy from
bs≥64).

The dispatch count is also higher than the hand count in §4: **~8,340 aten
calls per step**, not ~2,000. (This counts nested ops — `aten::linear` and the
`aten::mm` inside it are both counted — but the ratio to the ~57 ms of host
time is the number that matters: ~7 µs an op, the figure the doc assumes.)

This strengthens items 3 and 4 rather than changing them: de-syncing the loop
and capturing the step in a CUDA graph are the lever at bs≤64, and there is
more headroom under them than estimated.

**Unexplained:** one step in ten at bs=64 took 464.9 ms against a 57.0 ms
median. The build's `TORCH_CUDNN_SDPA_AVOID_RECOMPILE` warning text says a
bias forces cuDNN's recompiling path, and kv_len grows by one every step, so a
per-shape replan is the suspect — but one outlier in ten steps is not the
signature of a replan on every step. Not chased down.

---

## 5. This box is not the sweep's box

The harness reproduces the engine, and both are ~2.4× slower than the recorded
sweep:

| At bs=4, uniform-512x128 | ms/decode step |
|---|---|
| sweep record, 2026-08-19, commit `faa6722` clean | 23.6 |
| `bench_throughput` re-run today, same commit | **60.3** |
| `profile_decode` today | **57.0** |

`profile_decode` agrees with the engine to within 6%, so the tool is faithful
and the gap is environmental. Ruled out: code drift (no `src/` change since
`faa6722` — the two commits since touch docs, `benchmarks/report.py`, tests and
lockfiles),
`CUDA_LAUNCH_BLOCKING` (unset), CPU model and thread count (identical Xeon
8480+, 112 threads), CPU frequency (boosts to 3.8 GHz under load). Not ruled
out: a cgroup quota of 23.8 CPUs on a 224-thread box, a noisy neighbour
(loadavg ~6), or simply a different physical host behind the same SKU.

Since decode is 99% host-bound, a slower host scales ms/step almost one for
one, which is what the table shows. Today's MBU figures — 0.087 (bs=1), 0.112
(bs=64), 0.148 (bs=256) — are depressed for the same reason and should not be
compared with the sweep's 0.18–0.225.

Worth noting for its own sake: **decode's throughput on this repo is a function
of host speed**, so a benchmark record without a matching host is not
reproducible. `s_per_decode_step` moved 2.4× with nothing but the machine
underneath it.

---

## 6. Where this leaves the priority list

| | [`raising-decode-mbu.md` §5](raising-decode-mbu.md) said | Now |
|---|---|---|
| 1 | Profile a step; check the backend — *"decides everything below"* | **Done.** cuDNN, fused, GQA-native. The premise that decode runs unfused attention was wrong |
| 2 | KV view instead of gather, ~1.5× at bs=256 | **Promoted.** 40% of device time at bs=256, the largest measured waste. Verify no `contiguous` appears in its place |
| 3 | De-sync the decode loop, 10–25% | **Unchanged, better supported.** 99% host share at bs≤64 |
| 4 | CUDA graphs on the decode step, 2–4× at bs≤64 | **Unchanged, better supported.** ~8,340 dispatches a step against 10–17 ms of device work |
| 5 | `torch.compile` the decoder layer | Unchanged |
| — | — | **New:** assert the backend. Falling off cuDNN onto math costs 5.2× device time and nothing would announce it |

The one thing §5 offered as the cheap experiment — `sdpa_kernel([CUDNN_ATTENTION, MATH])` —
is already what runs. There is nothing to win there.

---

## 7. Reproducing

```bash
# The three answers, on one wave: backend, copies, host-vs-device
uv run python -m benchmarks.profile_decode --model Qwen/Qwen3-8B \
  --dtype bfloat16 --batch-size 64 --steps 10 --warmup 4

# What the source reading predicted, for comparison
uv run python -m benchmarks.profile_decode --model Qwen/Qwen3-8B \
  --dtype bfloat16 --batch-size 64 --steps 10 --warmup 4 --sdpa-backend math

# Price the engine's per-row finish bookkeeping (section 3)
uv run python -m benchmarks.profile_decode ... --engine-syncs
```

It writes a chrome trace to `benchmarks/traces/` by default (`--no-trace` to
skip), openable in `chrome://tracing` or perfetto.dev.
