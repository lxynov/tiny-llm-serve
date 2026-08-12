# Benchmarking plan

Goal: produce numbers that (a) are defensible, (b) teach you where inference time
actually goes, and (c) survive being re-run against continuous batching and paged
attention without redesign.

**Decisions up front**

| Question | Answer |
|---|---|
| Is the existing harness good enough? | The *protocol* is good, but the harness measures too little: no latency, no roofline, no sweep tooling. Fix before spending GPU money. |
| Is Qwen3-0.6B too small for H100? | Yes — at 0.6B you measure Python dispatch overhead, not the GPU. Use **Qwen3-8B** as the primary. Keep 0.6B/1.7B as a deliberate *contrast* experiment. |
| Does the code support larger models? | Yes for the **dense** Qwen3 family (0.6B–32B, all `Qwen3ForCausalLM`). No for MoE/quantized. 8B and 14B are safe; 32B has a host-RAM trap (§5). |
| Is a batch-size sweep an "ablation study"? | No — that's a **parameter sweep** producing a throughput–latency **Pareto curve**. Ablation = removing a component. You want both. See §9. |
| Budget | ~$20 of $25: ~2h on a cheap GPU to shake out CUDA, ~5h on H100 SXM for real runs, buffer for one re-run. See §6. |

---

## 1. Verdict on the existing harness

### What is already right — don't churn it

`README.md` §Benchmark rules 1–4 are better than what most hobby projects have,
and better than what a lot of production teams write down:

- **One variable per comparison** + refusing cross-machine comparisons.
- **Pre-tokenized prompts** (`generate_ids` taking `list[int]`) — tokenizer time
  never pollutes engine numbers.
- **`ignore_eos` + seeded workloads** — every mode processes *byte-identical*
  work, so a delta is attributable to the engine. This is the single most
  important design choice in the whole harness and it's already there.
- **Warm-up + repeats + mean-with-spread**, with a device barrier on both sides
  of the timed region, so the timer measures kernel execution rather than
  submission.
- **Both memory peaks** (`allocated` and `reserved`) — OOM is decided by
  reserved, so it's the one that predicts a sweep's capacity ceiling.
- **Records are data**, with commit + `dirty` + full environment. The `dirty`
  flag in particular is a detail people learn to add only after being burned.
- **`kv_efficiency`** — a metric chosen because it's the number paged attention
  will move. Forward-looking metric design.

Keep all of it. The changes below are additive.

### P1 gaps — the harness measures too little

- **No latency metrics at all.** Only aggregate throughput. TTFT / TPOT / ITL are
  the metrics continuous batching exists to improve; if you don't collect them
  now, your static-vs-continuous comparison later will have nothing to show.
- **Prefill and decode are not separated.** They are different machines: prefill
  is compute-bound (big GEMMs, high arithmetic intensity), decode is
  memory-bandwidth-bound (one token per sequence, weights re-read every step).
  A single `wall_time_s` averages two regimes and hides both.
- **No roofline.** `output_tok_s` alone can't tell you whether 300 tok/s is good.
  MBU/MFU (§2.3) convert it into "you are at 6% of the hardware limit", which is
  the number that makes the rest of the work obvious *and* is the number a
  reviewer of your resume project will find interesting.
- **No cost metric.** `$/1M tokens` is the metric the industry actually buys on.

### P2 — tooling that doesn't exist yet

`benchmarks/` is `bench_throughput.py` + `workloads.py`: one run per invocation,
one JSON record per run, and nothing that reads those records back. No sweep
driver, no aggregation, no plots. You cannot run a 40-point matrix by hand on a
metered GPU.

---

## 2. What to measure

### 2.1 Throughput (you have some of this)

| Metric | Formula | Note |
|---|---|---|
| `output_tok_s` | generated ÷ wall | The headline for a serving engine |
| `total_tok_s` | (prompt + generated) ÷ wall | Rewards prefill; report both |
| `request_tok_s` / `req_s` | requests ÷ wall | What a capacity planner uses |
| `cost_per_mtok_usd` | `price_hr / (output_tok_s × 3600) × 1e6` | Record `--price-per-hour`; it's the only metric that survives a hardware change |

### 2.2 Latency (missing — add)

For each request *i*, with submit time $t^{sub}_i$ (0 for all in an offline
drain; nonzero once you add arrivals in §3.4):

| Metric | Formula |
|---|---|
| **TTFT** (time to first token) | $t^{tok_1}_i - t^{sub}_i$ |
| **TPOT** (time per output token) | $(t^{end}_i - t^{tok_1}_i) / (\text{out}_i - 1)$ |
| **ITL** (inter-token latency) | the raw per-step gaps; report the *distribution*, not just the mean |
| **E2E** | $t^{end}_i - t^{sub}_i$ |

Report **p50 / p90 / p99** for each, not means. Tail latency is where static
batching's head-of-line blocking lives, and a mean hides it completely.

> Static batching prediction to verify: within one wave every sequence has
> roughly the *same* TTFT (they prefill together) and the *same* E2E (they all
> wait for the wave's longest output). So p99/p50 E2E will be near 1.0 *within a
> wave* but the ratio of E2E to a sequence's own required work will be terrible
> for short requests. Add a per-request `e2e_over_ideal = E2E_i / (own work)` —
> that ratio is precisely what continuous batching collapses toward 1.

**How to collect it minimally.** Don't refactor the engine yet. Add an optional
out-parameter that the generate methods fill with timestamps:

```python
@dataclass
class StepTrace:
    prefill_end: float = 0.0
    step_ends: list[float] = field(default_factory=list)  # one per decode step
```

`generate_batch_ids(..., trace: StepTrace | None = None)` appends
`time.perf_counter()` after each step (with a `torch.cuda.synchronize()` when
tracing is on — accept the ~2-5% probe overhead, and record `traced: true` so
traced and untraced records are never compared). From `step_ends` plus each
request's known `output_len`, the harness derives every metric above without the
engine knowing what a percentile is.

### 2.3 Roofline: MBU and MFU (missing — add; this is the insight generator)

**Memory Bandwidth Utilization** — the right lens for *decode*:

$$\text{MBU} = \frac{(\text{weight bytes} + \text{KV bytes read per step})/\text{step time}}{\text{peak HBM bandwidth}}$$

Weights are read once per step for the *whole batch*, so MBU rising with batch
size is exactly the batching win, expressed physically.

**Model FLOPs Utilization** — the right lens for *prefill*:

$$\text{MFU} = \frac{2 N_{\text{params}} T_{\text{prompt}} + 4 L s^2 d}{\text{prefill time} \times \text{peak bf16 FLOP/s}}$$

The $4Ls^2d$ attention term is ~2% at s=512 for 8B but grows as $s^2$ — keep it
in the formula so the 2048-token prompt row is honest.

**Hardware constants** (verify against your pod's actual SKU with `nvidia-smi`):

| GPU | HBM BW | bf16 dense |
|---|---|---|
| H100 SXM5 80GB | 3.35 TB/s | ~989 TFLOP/s |
| H100 PCIe 80GB | 2.0 TB/s | ~756 TFLOP/s |
| H100 NVL 94GB | 3.9 TB/s | ~989 TFLOP/s |

Put these in a small `benchmarks/hardware.py` table keyed by
`torch.cuda.get_device_properties().name`, falling back to `None` (emit MBU/MFU
as null rather than guessing).

### 2.4 Sanity anchors — memorize these before you rent anything

Model facts for this family (computed from the real configs; KV is bf16,
2 × layers × kv_heads × head_dim × 2 bytes per token):

| Model | Params | bf16 weights | KV / token | Weights == KV at | bs=1 decode ceiling on H100 SXM |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.60 B | 1.19 GB | 112 KiB | 10.4 K tokens | **2815 tok/s** |
| Qwen3-1.7B | 1.72 B | 3.44 GB | 112 KiB | 30.0 K tokens | **973 tok/s** |
| Qwen3-8B | 8.19 B | 16.38 GB | 144 KiB | 111 K tokens | **205 tok/s** |

Two things fall out of this table immediately:

1. **KV per token barely changes across the family** (8 KV heads × 128 head_dim
   at every size). So the smaller the model, the more *KV-dominated* it is:
   Qwen3-0.6B's cache equals its entire weight footprint after ~10K tokens.
   That's a genuinely interesting finding for the writeup and it argues that
   paged attention matters even at 0.6B.
2. **The bs=1 ceilings are the bogosity detector.** If a run reports >205 tok/s
   at bs=1 on Qwen3-8B, the measurement is wrong — the timed region is not
   covering the work you think it is. If it reports 40 tok/s, you're at 20% MBU and something (host
   overhead) owns the other 80%.

### 2.5 Memory

Add `kv_bytes_reserved` (computable exactly: `slots × max_model_len × 144 KiB`)
and `weight_bytes` next to the two peaks the record already carries
(`max_memory_allocated`, `max_memory_reserved`); keep `kv_efficiency`. Peak memory as a function of batch size is a first-class result
— it's the curve that shows *why* you need paged attention.

---

## 3. Workloads

### 3.1 What you have

The three existing workloads are well-chosen and each has a stated purpose
(`uniform-512x128` = best case, `mixed-out` = tail waste, `sharegpt-like` =
realism). Random token ids are fine — throughput is content-independent, and it
keeps the harness tokenizer-free. Keep all three.

### 3.2 How vLLM and SGLang do it

| | vLLM (`vllm bench serve`) | SGLang (`bench_serving.py`) |
|---|---|---|
| Datasets | `sharegpt`, `random`, `sonnet`, `burstgpt`, `hf`, `prefix_repetition`, multimodal variants | `sharegpt`, `random`, `generated-shared-prefix`, `random-ids` |
| Load shape | `--request-rate` (Poisson), `--burstiness` (gamma shape) | Poisson or uniform arrivals, or fixed concurrency |
| Headline metrics | throughput + TTFT/TPOT/ITL percentiles + **goodput under SLO** | same, plus kernel-level `bench_one_batch.py` |
| Offline variant | `benchmark_throughput.py` | `bench_offline_throughput.py` |

Three patterns worth stealing:

- **A parametric synthetic generator** (vLLM's `random`: `--input-len`,
  `--output-len`, `--range-ratio`) — the workhorse for controlled sweeps. Your
  three workloads are *fixed presets*; a sweep over sequence length needs a knob.
- **A shared-prefix workload** (SGLang's `generated-shared-prefix`: G groups × P
  prompts, each = shared system prompt + short question). This exists to make
  prefix caching visible. Define it now even though nothing uses it yet.
- **An arrival process.** Both tools drive load at a *rate*, not as a batch dump.
  This is the crux for your roadmap (§3.4).

### 3.3 Additions

Add to `workloads.py`:

```python
# Parametric — the sweep workhorse. Named "fixed:512x128" etc., parsed from the
# CLI so a sweep can walk the (in, out) grid without new preset names.
"fixed:{in}x{out}"        # deterministic, range_ratio=0
"random:{in}x{out}:{r}"   # lengths uniform in [(1-r)·L, (1+r)·L]

# Regime corners — expose the two machines separately.
"prefill-heavy"           # 2048 in / 16 out   -> compute-bound, MFU story
"decode-heavy"            # 32 in / 512 out    -> bandwidth-bound, MBU story

# Forward-looking: nothing exploits this yet, and that is the point. It is the
# baseline you will compare prefix caching against later.
"shared-prefix"           # G groups x P prompts: S shared tokens + Q unique
```

Also make records **self-describing**: emit prompt/output length mean, median,
p99 and the total token count into the record's `workload` block. Six months
later you will not remember what `mixed-out` seed 0 with N=64 actually contained.

**On real ShareGPT:** worth adding *once*, as a `--dataset-path` option that
reads real conversation lengths (ignore the text; you only need the length
distribution, and pre-tokenized random ids keep the tokenizer out of the loop).
It buys you one sentence — "our synthetic length distribution matches the real
trace's p50/p99 within X%" — which is exactly the kind of validation that
distinguishes a careful benchmark. It is P2, not a blocker.

### 3.4 The arrival process — design this now, use it later

Static batching cannot express arrivals, and **that is the finding**. Continuous
batching's win is *not* mostly raw throughput on a fixed drained set; it's
throughput at a given tail-latency SLO under a load that arrives over time.

Add `arrival_s: float` to `Request` now (0.0 for every offline workload), plus
`--request-rate` (Poisson, `inf` = offline drain) and `--burstiness`. In static
mode, a request that arrives mid-wave simply waits for the next wave — measure
that queueing delay and report it. That number is the "before" picture that
makes your continuous-batching commit worth writing.

---

## 4. Code changes, in dependency order

Everything in P0 and P1 is local work on your Mac against the tiny test
checkpoint. **Do not rent a GPU until P0+P1 are done and `pytest` is green** —
GPU time is the scarce resource, laptop time isn't.

### P0 — before renting anything

| # | Change | Where | Acceptance |
|---|---|---|---|
| 4.1 | **Slice the last hidden state before `lm_head`** | `qwen3.py:155`, `llm_engine.py:125-130` | See below — this is a blocker |

**On 4.1 — this is a hard blocker for the batch-size sweep, not a nicety.**
`Qwen3ForCausalLM.forward` applies `lm_head` to *every* prefill position, then
`llm_engine.py:130` throws all but the last row away. The logits tensor is
`batch × prompt_len × 151936`:

| batch (512-token prompts) | logits tensor, bf16 |
|---|---|
| 8 | 1.2 GB |
| 32 | **5.0 GB** |
| 64 | **10.0 GB** |
| 128 | **19.9 GB** |

At batch 128 that's 20 GB of pure waste on top of 16.4 GB weights + 11.5 GB KV,
and your sweep OOMs before reaching the interesting part of the curve. It also
burns ~8% of prefill FLOPs. Fix: pass the wanted row indices down (or slice
hidden states in the engine before calling `lm_head`) so prefill produces
`batch × 1 × vocab`. Cost drops to ~0.3 MB. **Measure before and after** — it's a
clean, quantified optimization for the writeup (§7, E6).

The engine already computes those row indices (`llm_engine.py:128-130` builds
`rows`/`lens` and gathers each row's last real token), so the change is moving an
existing selection earlier, not inventing one.

### P1 — the metrics that make it worth doing

| # | Change | Notes |
|---|---|---|
| 4.2 | `StepTrace` timestamps in both generate paths (§2.2) | Gate behind a `trace=` arg; record `traced: true` |
| 4.3 | Derive TTFT/TPOT/ITL/E2E + p50/p90/p99 in the harness | Engine stays percentile-free |
| 4.4 | Split `prefill_time_s` / `decode_time_s` / `decode_steps` / `s_per_decode_step` | Falls out of `StepTrace` |
| 4.5 | `weight_bytes`, `kv_bytes_reserved` in metrics | Needed by 4.6 |
| 4.6 | `hardware.py` peak-BW/FLOP table → `mbu`, `mfu` | Null when the GPU isn't in the table |
| 4.7 | `--price-per-hour` → `cost_per_mtok_usd` | Also record it in `environment` |
| 4.8 | Parametric + corner workloads (§3.3); workload stats in the record | |
| 4.9 | Record `allow_tf32`, SDPA backend, `nvidia-smi` clocks/power/temp in `environment` | See §10; today `environment()` stops at OS/python/torch/CPU/GPU |
| 4.10 | Warm-up hygiene: assert `wall_times_s[0]` is within CV of the median, else warn "insufficient warm-up" | Free diagnostic from data already recorded; see below |
| 4.11 | `arrival_s` on `Request`, `--request-rate`, `--burstiness` (§3.4) | Wire it now, exploit it after continuous batching |
| 4.12 | **Stop gathering the KV window every decode step** | `preallocated.py:148-151`; see §7.1. Quantify with E6 |
| 4.13 | `--wave-order {arrival,sorted}` in the harness | `waves()` is FIFO-only today; sorting by prompt+output length is a few lines and is all E7 needs |

**On 4.10 — what warm-up is and isn't allowed to prime.**
Warm-up exists to absorb *machine* state: CUDA context creation, first-launch
kernel loading, cuBLAS workspace allocation, cuBLAS/cuDNN shape autotuning, the
caching allocator's first `cudaMalloc` per size class, and the SM clock ramp off
idle. It must never prime the *measurement*.

Today the harness is safe on that point, and for a non-obvious reason worth
recording: each pass rebuilds its KV from scratch (`llm_engine.py:45` news a
`NaiveKVCache` per request, `llm_engine.py:110` news a `PreallocatedKVManager`
per wave), so no cached keys or values survive a pass. What *does* survive is the
allocator pool backing those tensors — which is exactly the thing worth priming.

This is also why the warm-up should stay a **full workload pass** rather than
something cheaper: `PreallocatedKVManager` sizes its allocation from
`wave_model_len(wave)`, so a smaller warm-up would allocate different-sized
blocks and fail to prime the size classes the timed passes actually use.

**The trap is in the roadmap.** Once prefix caching / RadixAttention lands,
state *does* persist across requests, and a warm-up running the identical
workload would leave every prompt cached — the timed passes would then measure a
100% cache-hit rate. The `repeats` loop has the same problem: pass 2 and 3 would
hit pass 1's cache. At that point you must either reset the cache between passes
or explicitly declare "steady state, warm cache" as the measured quantity and
label the record so it is never compared against a cold-cache one. Decide it
deliberately; do not inherit it by accident.

### P2 — tooling (write before the GPU session, it pays for itself in one run)

**`benchmarks/sweep.py`** — matrix driver. Requirements:

- Reads a small YAML/py matrix; **runs each point in a fresh subprocess** (a
  shared process carries allocator state across points and corrupts peak-memory
  numbers).
- **Catches `torch.cuda.OutOfMemoryError`, records `{"oom": true}`, continues.**
  Non-negotiable for a batch-size sweep — you *want* to find the ceiling, and a
  crash at point 6 of 40 wastes the whole session.
- **Resumable**: skip points whose record already exists. Pods die.
- Prints ETA and a running cost estimate.
- Optional per-point `nvidia-smi` sample appended to the record.

**`benchmarks/report.py`** — records → CSV + markdown tables + matplotlib PNGs.
The plots you actually need are in §8.

---

## 5. Model and hardware choice

### 5.1 Why Qwen3-0.6B on an H100 measures the wrong thing

Not "too small to be interesting" — **too small to be measuring the GPU**.

At bs=1, one decode step for Qwen3-0.6B must move 1.19 GB of weights: at
3.35 TB/s that's **0.36 ms**. Meanwhile the eager PyTorch step issues roughly
15-20 kernels per layer × 28 layers ≈ 500-600 dispatches, at ~5-10 µs of Python
+ ATen dispatch each ≈ **3-7 ms of host time**. The host is 10-20× the GPU. You
would be benchmarking your pod's CPU, and a community-cloud pod with 4 shared
vCPUs would "prove" the H100 is slow.

For Qwen3-8B the GPU floor is 4.9 ms/step, comparable to host overhead at bs=1
and dominant by bs≥8. That's the regime where batching effects are real.

**But run 0.6B deliberately, once** (E4). "Throughput is flat from bs=1 to bs=8
on 0.6B and MBU is 2%, because eager dispatch owns the step" is a *result*, and
it's the empirical motivation for CUDA graphs / `torch.compile` later.

### 5.2 Model support: what actually loads

`MODEL_REGISTRY` has exactly `Qwen3ForCausalLM`, and `models/__init__.py` already
documents the boundary correctly. So:

| Model | bf16 weights | Verdict |
|---|---|---|
| Qwen3-0.6B / 1.7B / 4B | 1.2 / 3.4 / 8 GB | ✅ contrast experiments |
| **Qwen3-8B** | 16.4 GB | ✅ **primary** — leaves ~60 GB for KV + logits + activations |
| Qwen3-14B | 29.5 GB | ✅ optional second scale point |
| Qwen3-32B | 65.5 GB | ⚠️ see below |
| Qwen3-30B-A3B and any `Qwen3Moe*` / `Qwen3Next*` | — | ❌ different architecture, not implemented |
| Any FP8/AWQ/GPTQ checkpoint | — | ❌ explicitly rejected at `loader.py:77-81` |

**The 32B trap:** `loader.load_weights` (`loader.py:51-61`) materializes *every*
tensor into a CPU dict before `load_weights` copies them to the GPU. For 32B
that's a 65 GB host-RAM spike on top of the GPU copy. You'd need a pod with
≥96 GB RAM and you'd pay several minutes of load time *per subprocess* in a
sweep. Not worth it at this budget. (If you ever want it: stream tensors
straight from `safe_open` to the device instead of building the dict — a good
small commit, but out of scope here.)

**Verified config facts for the primary model (Qwen3-8B):** 36 layers,
hidden 4096, 32 Q heads / 8 KV heads, head_dim 128, intermediate 12288,
vocab 151936, `tie_word_embeddings: false`, `max_position_embeddings` 40960.

### 5.3 GPU choice

H100 SXM is the right call for the headline: it's the common reference point and
its 3.35 TB/s HBM3 makes the decode-bandwidth story crisp. Prices move — check
before booking — but as of writing RunPod is roughly:

| GPU | ~$/hr (community) | Role |
|---|---|---|
| H100 SXM 80GB | ~$2.69 | **Headline runs** |
| H100 PCIe 80GB | ~$1.99 | Fallback (note: 2.0 TB/s, different roofline — do not mix records) |
| A100 SXM 80GB | ~$1.39 | Fallback |
| A40 48GB | ~$0.44 | **Dev / shakeout** — 8B bf16 fits fine |

Two pod-selection rules that matter more than they look:

- **Take ≥8 vCPUs.** Eager decode is partly host-bound; a 4-vCPU shared pod
  changes your numbers.
- **Never mix SKUs in one comparison.** H100 SXM and H100 PCIe differ by 1.7× in
  bandwidth. Rule 1 already says this; the failure mode here is that RunPod
  silently gives you whichever is available.

---

## 6. RunPod runbook and budget

### 6.1 Budget

| Phase | GPU | Hours | Cost |
|---|---|---|---|
| 0. Harness work (§4 P0+P1+P2) | none — your Mac | — | $0 |
| 1. Shakeout: CUDA correctness, calibrate timings, size the matrix | A40 @ ~$0.44 | 2.5 | ~$1.10 |
| 2. Headline runs E1–E8 | H100 SXM @ ~$2.69 | 5 | ~$13.45 |
| 3. Buffer: one re-run of whatever broke | H100 SXM | 2 | ~$5.40 |
| 4. Network volume (50 GB, prorated) | — | — | ~$1 |
| **Total** | | | **~$21** |

The single biggest cost saver is Phase 1: every bug you find at $0.44/hr is a bug
you don't find at $2.69/hr. The second is scripting Phase 2 so it's one
`bash run_all.sh` you launch and walk away from.

### 6.2 Pod setup

Create a **network volume** first (50 GB) and mount at `/workspace`, so a killed
pod doesn't cost you a re-download of the checkpoints.

```bash
# Template: RunPod PyTorch 2.x CUDA 12.x. Then:
export HF_HOME=/workspace/hf          # persist checkpoints on the volume
export TOKENIZERS_PARALLELISM=false   # avoid fork warnings skewing host time

cd /workspace && git clone <your repo> tiny-llm-serve && cd tiny-llm-serve
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync

# Pre-download once, outside any timed region (16.4 GB, a few minutes)
uv run python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-8B')"

# Provenance: capture this into the results directory before anything else
nvidia-smi -q > benchmarks/results/h100/nvidia-smi-before.txt
git rev-parse HEAD && git status --porcelain   # MUST be clean
```

### 6.3 Session discipline

1. **`uv run pytest`** — a broken build discovered at $2.69/hr is embarrassing.
   This is also the correctness gate: `tests/backends.py` parametrizes over
   `cuda` when it's there, so the existing static-vs-sequential greedy parity
   check (`tests/test_llm_engine.py`) runs on the GPU here for the first time.
   A fast engine that emits wrong tokens is worth nothing, and this is the class
   of batching bug that only appears on CUDA.
2. **One tiny calibration run** (`--num-requests 4 --repeats 1`) and check it
   against the §2.4 anchors before launching the matrix. The parity test runs on
   a tiny random checkpoint, so also eyeball one real-model completion —
   `python -m tiny_llm_serve.engine.llm_engine` — as the "the 8B weights loaded
   correctly" check.
3. Launch the sweep under `tmux` (pods drop SSH), with output tee'd to the
   network volume.
4. **Re-run E1 at the very end of the session.** If it doesn't match the start
   within CV, the pod drifted (thermal throttling, noisy neighbor) and you know
   it before you publish.
5. `git add benchmarks/results/h100 && git commit` **before terminating the pod.**
   Do this. Terminated pods do not give the data back.

---

## 7. Experiment matrix

All on Qwen3-8B / bf16 / greedy / H100 SXM unless stated. Time estimates assume
eager decode at ~8-15 ms/step — **calibrate them in Phase 1** and resize.

| ID | Experiment | Configuration | Est. | What it answers |
|---|---|---|---|---|
| **E1** | **Baseline ablation** | `sequential`, then `static` at bs=1 and bs=8. `uniform-512x128`, N=32 | 20 m | Decomposes the speedup into "not O(n²) anymore" (`sequential` → `static` bs=1) vs "actually batching" (bs=1 → bs=8). Run again at session end as the drift check. |
| **E2** | **Batch-size sweep** (headline) | bs ∈ {1,2,4,8,16,32,64,128} × {`uniform-512x128`, `mixed-out`, `sharegpt-like`}, N=64 | 90 m | The throughput–latency **Pareto curve**; where MBU saturates; where peak memory ends the party. |
| **E3** | **Sequence-length sweep** | in ∈ {128,512,2048} × out ∈ {32,128,512}, bs=16, N=32 | 25 m | Separates the prefill (compute-bound) and decode (bandwidth-bound) regimes. Pairs with `prefill-heavy` / `decode-heavy` corners. |
| **E4** | **Model-scale contrast** | Qwen3-{0.6B, 1.7B, 8B} × bs ∈ {1,8,32}, `uniform-512x128` | 35 m | The §5.1 claim, empirically: flat throughput + ~2% MBU at 0.6B = host-bound; rising MBU at 8B = GPU-bound. |
| **E5** | **dtype ablation** | bf16 vs fp32, bs ∈ {1,16} | 15 m | Confirms decode is bandwidth-bound (fp32 ≈ 2× the bytes ⇒ ≈ 2× the step time) and prefill is compute-bound (fp32 has no tensor-core path ⇒ far worse than 2×). Set `allow_tf32` explicitly and record it. |
| **E6** | **Optimization ablations** | E2 subset (bs ∈ {8,32,64}) before vs after change 4.1 (lm_head slice) and change 4.12 (KV gather, §7.1) | 35 m | Two quantified, self-found optimizations. Run each as a separate commit; the records carry the SHA. |
| **E7** | **Scheduling ablation** | `mixed-out`, bs=16: FIFO waves vs length-**sorted** waves (needs 4.13) | 15 m | How much of static batching's loss is a *scheduling* problem vs a *mechanism* problem. This is the direct motivation for continuous batching, and it's cheap. |
| **E8** | **Capacity** | max bs that fits, from E2's OOM points | free | Peak memory vs bs; `kv_efficiency` vs workload — the case for paged attention. |

Total ≈ 4 GPU-hours + ~1h of setup/loads/mistakes ⇒ the 5h Phase 2 budget.

**No third engine mode.** The middle term of E1 — preallocated KV without
batching — is `static` at `--batch-size 1`, which the existing `--mode static`
already covers; a `sequential-prealloc` mode would be the same code path behind
a second name. E1's two static points also fall out of E2's bs sweep, so the only
run unique to E1 is `sequential` (run it at E2's N so the rows are comparable).

### 7.1 A second optimization the benchmark should surface

`preallocated.py:148-151` returns
`k_cache[self._slot_ids, : self._kv_len]` — `_slot_ids` is a **tensor**, so this
is advanced indexing, which allocates and gathers a fresh tensor rather than
returning a view. (Verified: the result's `data_ptr()` differs from the cache's.)
That happens **for both K and V, in every layer, on every decode step** — an
extra full read + write of the attention window on top of the read attention
already has to do.

The slot pool is `list(range(num_slots))`, so a static wave's slots are exactly
`{0 … num_slots-1}` and a basic slice `k_cache[:num_slots, :kv_len]` *is* a view.
One catch to handle in the fix: `admit` pops from the **end** of `_free_slots`,
so a full wave receives its slots in descending order and slicing without
changing that would silently transpose batch rows against cache rows. Hand them
out in ascending order (or have the engine ask for a contiguous range), then
slice. Fix it, re-run E6, and the delta is your MBU improvement. This is exactly
the kind of thing a roofline metric finds and a tok/s number hides — a good story
for the writeup.

### 7.2 Expected shapes (write these down before you run — then check)

Predicting results first turns a surprise into a finding instead of a shrug.

- **E2, throughput vs bs:** near-linear at small bs (host-overhead-dominated —
  the step time barely moves), bending over as bs grows, plateauing when decode
  becomes genuinely bandwidth-bound. **The knee is the interesting point**, and
  its location is the headline claim.
- **E2, step time vs bs:** roughly *flat* while overhead-bound, then rising
  linearly. Where it starts rising is the same knee. This is the cleanest
  single diagnostic in the whole plan.
- **E2, p99 E2E latency vs bs:** rises monotonically. Overlaid on throughput,
  this is the Pareto plot.
- **E1:** expect the `sequential` → `static` bs=1 gap to be *large* at 512+128
  tokens; the naive cache's `torch.cat` (`naive.py:32-33`) re-copies the whole
  cache on all 128 decode steps, while the preallocated slot writes in place.
- **E7:** length-sorted waves should recover a solid fraction of the `mixed-out`
  loss without any new mechanism — which is the honest framing of how much
  continuous batching's win is scheduling rather than machinery.

---

## 8. Reporting

Five plots and three tables. Nothing else.

**Plots** (`report.py`): (1) throughput vs batch size, one line per workload;
(2) throughput vs p99 latency — the Pareto frontier, points labelled by bs;
(3) MBU vs batch size with the 100% roofline drawn in; (4) peak memory and
`kv_efficiency` vs batch size; (5) E4's per-model throughput-vs-bs, showing the
flat 0.6B line next to the rising 8B one.

**Tables** for the README: the E1 ablation decomposition; the metric definitions
(extend the existing table); the "what changed" table from E6.

**Rule to keep:** every number in the README traces to a committed JSON record
with a clean-tree commit SHA. The existing Rule 4 already says this — hold the
line even when a `dirty: true` run looks better.

---

## 9. Terminology (you asked, and precision here is cheap)

| Term | Means | Your case |
|---|---|---|
| **Ablation study** | Remove/disable one component; the delta is that component's contribution | E1 (naive cache vs preallocated at bs=1), E6 (with/without the lm_head slice), E7 (FIFO vs sorted scheduling) |
| **Parameter sweep** / sensitivity analysis | Vary a knob over a range; report the curve | E2 (batch size), E3 (sequence length) — **this is what you described**, not an ablation |
| **Scaling study** | Sweep over problem or model size | E4 |
| **Roofline analysis** | Compare measured to hardware ceiling | MBU / MFU |
| **Pareto frontier** | The non-dominated set in a 2-objective tradeoff | Throughput vs p99 latency as bs varies — *the* canonical inference-engine plot |

Industry uses "ablation" loosely for both, but a plan that distinguishes them
reads as someone who has designed experiments rather than just run them.

---

## 10. Pitfalls checklist

- [ ] **Fresh subprocess per sweep point** — allocator state leaks across points.
- [ ] **TF32 is off by default for matmul** in recent torch. On an fp32 run
      that means true fp32 (~67 TFLOP/s on H100 vs ~989 bf16) — a 15× cliff.
      Set it explicitly and record the setting, or E5 is uninterpretable.
- [ ] **The boolean `attn_mask` on the decode path may disable SDPA's flash
      backend** (flash takes `is_causal`, not an arbitrary mask; masks fall back
      to the mem-efficient/math kernel). Record which backend actually ran
      (`torch.nn.attention.sdpa_kernel`, or check with the profiler). This could
      be a large, silent decode cost — and it's a candidate optimization.
- [ ] **Thermal/power throttling.** Sample `nvidia-smi --query-gpu=clocks.sm,
      temperature.gpu,power.draw,power.limit` per point. A sweep that runs 40
      minutes will throttle, and the late points will look worse for no reason.
- [ ] **Noisy neighbors on community pods** affect *host* time, which is a large
      share of eager decode. Take ≥8 vCPUs, and trust the CV guard.
- [ ] **`ignore_eos` must stay on** for every timed run, or modes stop doing
      identical work and Rule 1 is dead.
- [ ] **`dirty: true` ⇒ scratch run.** Commit before the headline sweep.
- [ ] **First CUDA call is slow** (context init, kernel autotuning, cuBLAS
      workspace). The warm-up pass covers it — never set `--warmup 0` on GPU.
- [ ] **Don't compare H100 SXM to H100 PCIe records.** Different rooflines.
- [ ] **Commit the results before terminating the pod.**

---

## 11. Forward compatibility

Design choices in this plan that exist so the *next* two commits are cheap:

| Now | Pays off at |
|---|---|
| `arrival_s` + `--request-rate` + `--burstiness` (4.11) | Continuous batching — its real win is tail latency under arriving load, which an offline drain cannot show |
| TTFT / TPOT / p99 (4.2-4.3) | Continuous batching — the metrics it moves |
| `e2e_over_ideal` per request (§2.2) | Continuous batching — collapses toward 1.0; the single cleanest before/after number |
| `kv_efficiency` (already there) + `kv_bytes_reserved` | Paged attention — the metric it exists to fix |
| `shared-prefix` workload (§3.3) | Prefix caching / RadixAttention |
| Warm-up/repeat cache semantics written down now (4.10) | Prefix caching — the pass that primes the machine must not also prime the cache, or every timed pass reports a 100% hit rate |
| MBU/MFU roofline (4.6) | Everything — it's the only metric that says how much room is left |

### Suggested commit sequence

1. `[Perf] Compute lm_head only for sampled positions` (4.1) ← quantify it
2. `[Bench] Record per-request latency, prefill/decode split, and roofline utilization` (4.2-4.7)
3. `[Bench] Add parametric, regime-corner, and shared-prefix workloads` (4.8)
4. `[Bench] Order static waves by length as a scheduling knob` (4.13)
5. `[Bench] Record run conditions and warn on insufficient warm-up` (4.9-4.10)
6. `[Bench] Add a resumable sweep driver and report generator` (P2)
7. `[Bench] Add arrival timestamps and request-rate load shaping` (4.11)
8. → rent the A40, then the H100
9. `[Perf] Avoid gathering the KV window on every decode step` (4.12, §7.1) ← quantify it
10. `[Docs] Add benchmark results` — plots + tables + committed records
