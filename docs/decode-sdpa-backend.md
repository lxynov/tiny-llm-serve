# Which SDPA Backend Decode Uses

A follow-up to [`raising-decode-mbu.md`](raising-decode-mbu.md), whose §5 makes
one check the prerequisite for everything else: *which backend does
`F.scaled_dot_product_attention` pick in decode?* This document answers it.

> **Corrected by measurement.** Decode gets **cuDNN**, not math. The two
> refusals below (§2) are confirmed on the box; the conclusion drawn from the
> priority order is not — cuDNN is promoted ahead of math by default, whatever
> `_get_sdp_priority_order()` reports. Read
> [`profiling-decode-on-h100.md`](profiling-decode-on-h100.md) for the trace,
> and treat §4 and §5 here as the counterfactual: what math *would* cost, which
> is 5.2× cuDNN's device time. Prefill gets FlashAttention, as stated.

Everything below is read off this repo's source and the PyTorch build in
`.venv` (`2.13.0+cu130`, git `cf30153c`), and was written before any of it was
run on a GPU — [§6](#6-confirming-it-on-the-box) is the check that turned it
into a measurement, and overturned its headline.

---

## 1. The call decode makes

[`layers/attention.py:45`](../src/tiny_llm_serve/layers/attention.py) passes
`enable_gqa=True` unconditionally, and decode always supplies a boolean mask:
[`qwen3.py:71-75`](../src/tiny_llm_serve/models/qwen3.py) takes it from
[`preallocated.py:156`](../src/tiny_llm_serve/kv/preallocated.py), shape
`[batch, 1, 1, kv_len]`.

So for `Qwen/Qwen3-8B` in bfloat16 the parameters SDPA selects on are:

| | |
|---|---|
| `query` | `[bs, 32, 1, 128]` |
| `key`, `value` | `[bs, 8, kv_len, 128]` |
| `attn_mask` | `[bs, 1, 1, kv_len]`, **bool, non-null** |
| `enable_gqa` | **`True`**, with 32 query heads over 8 kv heads |
| `is_causal` | `False` (`q_len == 1`) |

Those last two lines are each independently fatal to a fused kernel.

---

## 2. Why every fused backend is disqualified

| Backend | Verdict | Evidence |
|---|---|---|
| FlashAttention | out, on the mask | `sdp::check_for_attn_mask` — the string is in `libtorch_cuda.so`: *"Flash Attention does not support non-null attn_mask."* |
| mem-efficient | out, on GQA | `sdp_utils_cpp.h:430` opens the GQA gate only `if (params.enable_gqa && supports_gqa)`, and mem-efficient is the `supports_gqa=false` instantiation. With 32 ≠ 8 heads it then fails the same-heads check: *"For dense input, both fused kernels require query, key and value to have the same num_heads."* Corroborated by the wheel's own docs (`torch/nn/functional.py:6447`): GQA *"currently works only for Flash_attention and math kernel on CUDA tensor"* |
| cuDNN | **selected** | It is the `<supports_gqa=true, requires_same_num_heads=false>` instantiation, and `_scaled_dot_product_cudnn_attention` accepts an `attn_bias` — so it serves this call, and does |
| math | not reached | ~~the fallback everything else falls back to~~ |

The cuDNN row was originally written as *"eligible in principle, unreachable by
default"*, reasoning that `ATen/Context.h:481-486` puts math ahead of it. That
inference was wrong: cuDNN is promoted above math at selection time unless
`TORCH_CUDNN_SDPA_DEPRIORITIZED=1` says otherwise, so the only fused kernel
that can handle this call is the one that gets it.

---

## 3. Prefill is not affected

Prefill passes `attn_mask=None` with `is_causal=True` —
[`preallocated.py:94-100`](../src/tiny_llm_serve/kv/preallocated.py) explains
why right padding needs no mask — and flash supports GQA. So prefill is fused
and decode is not.

That asymmetry is invisible in the sweep because the two phases are timed
separately and only decode is scored by MBU. It also means the repo has *not*
been running "eager PyTorch attention" throughout, as the baseline doc's
framing implies — it has been running one fused phase and one unfused one.

---

## 4. What the math backend costs — and a bound that says the story is incomplete

Two things happen on the math path that do not happen on a fused one:

1. **The bool mask is materialized as a float bias** (`convert_boolean_attn_mask`
   → `masked_fill` with `-inf`), then broadcast-added to the scores.
2. **k and v are expanded from 8 heads to 32** — `_scaled_dot_product_attention_math`
   applies `repeat_interleave` along the head dim when `enable_gqa` is set,
   rather than letting a kernel broadcast the kv head across its query group.

Plus the obvious: no fusion. Scores, mask add, softmax and the second matmul
each round-trip a `[bs, 32, 1, kv_len]` tensor through HBM.

If the expansion fully materializes, the per-layer, per-tensor traffic at
bs=256 and kv_len=576 (bf16 → **X = 302 MB** of true KV) is:

| Stage | Bytes |
|---|---|
| gather ([`preallocated.py:177`](../src/tiny_llm_serve/kv/preallocated.py)) — read + write | 2X |
| `repeat_interleave` — read X, write 4X | 5X |
| matmul reads the 32-head copy | 4X |
| **total** | **11X** |

**But that cannot be what is happening.** The sweep's own numbers cap it:

- reported MBU 0.182 against 38.1 GB charged ⇒ **62.5 ms/step**
- 62.5 ms at the H100's 3.35 TB/s ⇒ **≤ 209 GB** actually moved, at *100%* of peak
- minus 16.4 GB of weights ⇒ **≤ 8.9×** the 21.7 GB KV minimum — and at a
  realistic 70–80% of peak, ≤ 6–7×
- the 11× model needs 256 GB in 62.5 ms, i.e. **4.1 TB/s**, which the hardware
  does not have

So at least part of the expansion is elided or served from L2 — most likely the
matmul broadcasts the kv head across its four query heads (each kv slice is
147 KB at these shapes, small enough to stay resident while the group consumes
it), leaving the doc's original 3× gather story roughly intact. The arithmetic
does not settle which; the trace does, in one line: look for
`aten::repeat_interleave` in a decode step and read its self CUDA time.

This is worth stating plainly because it cuts the other way too — §2 of the
baseline doc credits the gather with 3× and §5 speculates the backend adds "another
4×", and those two claims **cannot both be true** at the measured step time.

---

## 5. What this means for the priority list

> Superseded by [`profiling-decode-on-h100.md` §6](profiling-decode-on-h100.md).
> Decode is already fused, so the recommendation below — pin cuDNN — is a
> no-op, and the KV view moves from "worth a fraction" to the largest
> device-side item there is (40% of decode's device time at bs=256).

**Item 2 (KV view instead of gather) does not restore a fused kernel.** Even
with a perfect view, the bool mask still disqualifies flash and `enable_gqa`
still disqualifies mem-efficient. (True, and beside the point: cuDNN takes the
call as it stands.)

**The cheapest experiment is one line**, and it should come first:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
    ...  # decode
```

cuDNN handles both the 4:1 head ratio and the mask-as-bias, so on H100 (sm90)
this may simply work, with no change to the cache, the mask, or the engine.
Math stays in the list as the fallback if cuDNN rejects the shapes.

Beyond that, the real fixes remove the *reason* for the mask rather than
working around it:

| Option | Why it applies |
|---|---|
| `torch.nn.attention.varlen` (ships in 2.13) | GQA-aware, takes `cu_seqlens` instead of a mask — decode's per-row valid lengths are exactly what it wants |
| `flex_attention` with a block mask | fused, GQA-aware, and the mask becomes a compiled predicate rather than a materialized tensor |
| Equalize `kv_len` across the batch | with no ragged lengths there is no mask, and flash takes the call as-is |

---

## 6. Confirming it on the box

Ask PyTorch directly, with decode's shapes — `debug=True` prints each
backend's rejection reason:

```python
import torch
from torch.backends.cuda import (
    SDPAParams, can_use_flash_attention, can_use_efficient_attention,
    can_use_cudnn_attention,
)

b, s, d = 8, 576, 128
q = torch.randn(b, 32, 1, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, 8, s, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn_like(k)
mask = torch.ones(b, 1, 1, s, dtype=torch.bool, device="cuda")
p = SDPAParams(q, k, v, mask, 0.0, False, True)  # last arg = enable_gqa

print(can_use_flash_attention(p, True))
print(can_use_efficient_attention(p, True))
print(can_use_cudnn_attention(p, True))
```

A harder check against the real engine — this *raises* if decode cannot use a
fused kernel:

```python
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    llm.generate_batch_ids(...)
```

And in a profiler trace, the tell is which op appears in the decode step:

| Op in the trace | Means | What the trace showed |
|---|---|---|
| `aten::_scaled_dot_product_attention_math` | math — this document's claim | absent |
| `aten::repeat_interleave` (self CUDA time, and how much) | settles §4's open question | absent — no GQA expansion happens |
| `aten::_scaled_dot_product_flash_attention` | flash — expected in **prefill only** | prefill only, as expected |
| `aten::_scaled_dot_product_efficient_attention` / `_cudnn_attention` | would contradict §2 | `_cudnn_attention`, 36 calls a step — see the banner at the top |

Both checks are packaged in
[`benchmarks/profile_decode.py`](../benchmarks/profile_decode.py), which runs
the eligibility probe and the trace over one wave of a real workload.
