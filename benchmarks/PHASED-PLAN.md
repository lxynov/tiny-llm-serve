# Phased benchmarking plan

`PLAN.md` is the maximal version. This document is the route through it, cut
down to what a personal learning project should actually build, phase by phase,
each phase keyed to an engine feature that already exists.

**Phase 1 — the only phase specified in detail here — answers one question:**

> Under static batching, how do throughput, decode step time, and peak memory
> vary with batch size, and what physically limits each one?

| Phase | Engine feature it follows | Question it adds |
|---|---|---|
| **1** | static batching *(you are here)* | Throughput / step time / memory vs batch size, against the hardware roofline |
| **2** | continuous batching | Latency under *arriving* load |
| **3** | paged attention | Capacity: how many sequences fit, how much reservation is wasted |
| **4** | prefix caching | Cache hit rate and its effect on TTFT |

The scoping rule for Phase 1: build it only if leaving it out would make the
batch-size curve **wrong**, **unexplainable**, or **unrepeatable**. Everything
else goes in §3, with the phase that earns it back.

---

## 1. What is already done

- **The `lm_head` slice** (`PLAN.md` §4.1) landed in `f4b40c7`. Prefill now
  produces `batch × 1 × vocab` instead of `batch × prompt_len × vocab`. This was
  the hard blocker — the old path allocated ~20 GB of discarded logits at
  bs=128, so the sweep would have OOM'd before reaching the interesting region.
- **The measurement protocol** (README rules 1-4): one variable per comparison,
  pre-tokenized prompts, `ignore_eos`, warm-up + repeats + mean-with-spread,
  device barriers around the timed region, records carrying commit + `dirty` +
  environment. Unchanged.
- **Warm-up semantics are correct**, because every pass rebuilds its KV from
  scratch (`llm_engine.py:112` news a `PreallocatedKVManager` per wave). What
  the warm-up primes is the caching allocator, which is what it should prime.
  **This assumption breaks in Phase 4** — see §7.

---

## 2. Phase 1: what to build

Six changes. All of it is laptop work against the tiny test checkpoint; get
`pytest` green on all of it before renting a GPU.

### C1. Reject workloads that don't divide into whole waves

`waves()` (`bench_throughput.py:47-48`) chunks with a ragged tail, so
`--num-requests 64 --batch-size 128` runs **one wave of 64** and files it under
`batch_size: 128`. That is a wrong number, not a rough one. (`PLAN.md` §7's E2
has this bug as written.)

Error out when `num_requests % batch_size != 0`, and record the actual
`wave_sizes` list either way. Then pick N as a multiple of the largest batch
size in the sweep — **N=128 for bs ≤ 128**.

Holding N *identical* across every point (rather than N = k × bs) is what makes
README rule 1 true here: every point processes byte-identical work, so the delta
is attributable to batch size alone.

### C2. Split prefill from decode

A single `wall_time_s` averages two different machines — prefill is compute-bound
(one big GEMM over all prompt tokens), decode is memory-bandwidth-bound (all
weights re-read to produce one token per sequence). The batch-size story is
entirely about the second, and the mean hides it.

**You do not need per-step timestamps for this.** `PLAN.md` §2.2 proposes a
`StepTrace` list with a sync after every decode step; that exists to get an ITL
*distribution*, which Phase 1 doesn't use. Three timestamps per wave suffice:

```
sync; t0
prefill
sync; t1
decode loop            # decode_steps is already known from the workload
sync; t2
```

So: a small out-parameter on `generate_batch_ids` carrying two floats. Two extra
syncs per wave instead of one per step — negligible overhead, no `traced` flag,
no traced/untraced record split.

| New metric | Why |
|---|---|
| `prefill_time_s`, `decode_time_s` | The two regimes, separated |
| `decode_steps`, `s_per_decode_step` | **The cleanest diagnostic in the sweep**: flat while host-overhead-bound, rising once bandwidth-bound. The elbow between them is the headline |

*Expect this to surface something:* `run_static` builds a fresh
`PreallocatedKVManager` per wave (`bench_throughput.py:63`), which
`torch.zeros`-es the whole KV pool — 11.8 GB at bs=128 for 8B — inside the timed
region. It will show up as inflated prefill time. Whether to hoist it is a
Phase 1 *finding*, not a Phase 1 assumption.

### C3. Weight and KV byte accounting

- `weight_bytes` — sum over the loaded parameters.
- `kv_bytes_reserved` — exact arithmetic: `slots × max_model_len ×
  kv_bytes_per_token`, where `kv_bytes_per_token = 2 × layers × kv_heads ×
  head_dim × 2` (144 KiB/token across the whole dense Qwen3 family).

Peak memory vs batch size is a first-class Phase 1 result: it's the curve that
ends the sweep, and it's the case for paged attention in Phase 3.

### C4. MBU — the roofline

`output_tok_s` alone can't tell you whether 300 tok/s is good. Memory Bandwidth
Utilization converts it into "you are at 6% of what this GPU can physically do",
which is the number that makes everything else obvious.

```
bytes_per_step ≈ weight_bytes + batch × mean_kv_len × kv_bytes_per_token
mbu            = (bytes_per_step / s_per_decode_step) / peak_hbm_bandwidth
```

Add `benchmarks/hardware.py`: a small table of peak HBM bandwidth keyed on
`torch.cuda.get_device_properties().name` (H100 SXM5 = 3.35 TB/s, H100 PCIe =
2.0 TB/s, A100 SXM = 2.0 TB/s, A40 = 0.7 TB/s). Emit `null` for an unknown GPU —
never guess.

**MBU rising with batch size *is* the batching win, stated physically.** The
weights get read once per step no matter how many sequences share that step, so
more sequences means more useful work per byte moved. That sentence is the
thesis of the whole sweep.

### C5. Self-describing records

Two small additions so a record is readable a year later:

- Workload stats in the `workload` block: prompt/output length mean, median, max,
  and total token count.
- `"load": "offline-drain"`. Phase 2 will emit latency fields with *identical
  names* measured under arriving load; this one string is what keeps the two from
  being compared silently.

### C6. Sweep driver and report

**`benchmarks/sweep.py`** — a ~50-line subprocess loop, not a framework:

- **One fresh subprocess per point.** A shared process carries allocator state
  between points, which corrupts `max_memory_reserved` — the metric that defines
  the sweep's ceiling.
- **Skip points whose record already exists** (pods die mid-sweep).
- **Survive OOM**: a point that dies leaves no record and the sweep continues.
  Finding the ceiling *is* a result.

**`benchmarks/report.py`** — records → one CSV + three plots (§6). Compute
`$/1M tokens` here from a `--price-per-hour` argument rather than storing it in
records; it's one line of arithmetic and doesn't belong in the measurement.

### Not code, but blocking: use Qwen3-8B in bf16

At 0.6B, one bs=1 decode step moves 1.19 GB of weights — 0.36 ms of GPU work
against ~3-7 ms of eager Python/ATen dispatch. You'd be benchmarking your pod's
CPU. At 8B the GPU floor is 4.9 ms and dominates from bs≥8, which is where
batching effects are real. (`--dtype` defaults to `float32`; pass `bfloat16` on
GPU.)

Run 0.6B **once, deliberately**, as the contrast (E4 in §6) — "throughput is flat
from bs=1 to bs=8 and MBU is 2%, because dispatch owns the step" is a result.

---

## 3. Deliberately cut from Phase 1

Each of these is in `PLAN.md` and each is genuinely useful somewhere. None of
them is useful *here*, and a personal project's scarcest resource is your
attention.

| Cut | Why it doesn't earn Phase 1 | Comes back in |
|---|---|---|
| **TTFT / E2E per request** | Under static batching these are *redundant*: every sequence in a wave shares one prefill and ends when the wave ends, so TTFT ≡ prefill time and E2E ≡ wave duration. You'd be renaming C2's numbers | Phase 2, where they stop being redundant |
| **p50/p90/p99 latency** | A wave produces one independent observation, not B. At bs=128 with N=128 that's a "p99" over a single sample | Phase 2 (with enough arrivals to have a real sample) |
| **`e2e_over_ideal`** | Needs a per-request solo-runtime baseline to divide by — a second set of runs for a ratio `kv_efficiency` already gestures at | Phase 2, as the headline before/after |
| **ITL distribution / per-step `StepTrace`** | The mean step time is the diagnostic; the distribution costs a sync per step to answer a question you don't have yet | Whenever a stall needs explaining |
| **MFU (prefill roofline)** | A second formula, with a fiddly `4Ls²d` attention term, for the regime a *batch-size* sweep isn't about | Phase 1.5, if you sweep sequence length |
| **`nvidia-smi` clock/power/temp sampling** | Real, but it's ops instrumentation. The `wall_time_cv` guard already catches a throttled run as excess spread | If CV starts firing and you need to know why |
| **TF32 / SDPA backend recording** | TF32 only affects fp32 matmuls, and you're running bf16. Backend detection is fiddly | Phase 1.5, if you do the dtype ablation |
| **Arrival process, `--request-rate`, `--burstiness`** | Static batching has no mechanism that could respond to an arrival | Phase 2 — it's the *point* of Phase 2 |
| **`shared-prefix` workload** | Nothing in the engine exploits prefix sharing | Phase 4 |
| **Parametric workloads (`fixed:`/`random:`)** | Three presets cover a batch-size sweep; a length knob is for a length sweep | Phase 1.5 |
| **`--wave-order sorted`** | A scheduling ablation, not a batch-size result — and it's the best possible *opener* for Phase 2 | Phase 2 |
| **Real ShareGPT trace lengths** | Buys realism you can't act on yet. If you do add it: sample `(input, output)` as **pairs**, since all three current workloads have zero correlation between them by construction | Phase 2+ |

---

## 4. Two traps that actually bite

**Wave composition confounds the memory curve.** Each wave reserves
`max(prompt) + max(output)` *over that wave* (`bench_throughput.py:52`). Changing
batch size regroups requests, so `kv_efficiency` and peak memory move for two
reasons at once: the batch size, and who landed together.

For `uniform-512x128` this vanishes — every wave reserves 640 regardless of bs,
so the curve is clean. For `mixed-out` and `sharegpt-like` the confound is real,
but it is *also the actual phenomenon* (grouping heterogeneous requests badly is
what static batching does). Use uniform as the **clean** curve and the other two
as the **realistic** one, and say which is which.

**Latency and throughput are the same number here.** An offline drain is a
closed-loop system with concurrency pinned at B, so by Little's Law
`latency = B / throughput`. A throughput-vs-latency plot from a drain is the
offline batch-size tradeoff, not the open-loop serving curve vLLM and SGLang
publish — they look identical and answer different questions. This is the real
reason §3 cuts the latency metrics rather than merely deferring them: they'd
carry no information the throughput column doesn't already have.

---

## 5. Commit sequence

Nine steps. C1 first because it's a correctness bug in the measurement itself;
C2-C5 are independent record fields in rising order of how much new machinery
each needs; tooling last, once the record schema has stopped moving. Each commit
should be green on `pytest` on the Mac with the tiny checkpoint, and each one
that adds a metric should update the README metrics table in the same commit.

| # | Commit | Contains | Test to add |
|---|---|---|---|
| 1 | `[Fix] Reject workloads that do not divide into whole waves` | C1: the guard, plus `wave_sizes` in the record | `--num-requests 6 --batch-size 4` raises; `wave_sizes` matches |
| 2 | `[Feature] Time prefill and decode separately in batched generation` | C2: engine out-parameter + `prefill_time_s`, `decode_time_s`, `decode_steps`, `s_per_decode_step` | The two sum to roughly `wall_time_s`; `decode_steps` equals the expected step count |
| 3 | `[Feature] Record model weight and KV reservation sizes` | C3 | `kv_bytes_reserved` matches the closed-form arithmetic for the tiny config |
| 4 | `[Feature] Report memory bandwidth utilization against the device peak` | C4: `hardware.py` + `mbu` | Known bytes ÷ known time ÷ known peak gives the expected ratio; unknown GPU → `None` |
| 5 | `[Feature] Describe the workload and load shape in each record` | C5 | Length stats match a hand-computed workload |
| 6 | `[Feature] Add a resumable batch-size sweep driver` | C6a: `sweep.py` | An existing record is skipped; a failing point doesn't abort the sweep |
| 7 | `[Feature] Summarize benchmark records as a CSV and plots` | C6b: `report.py` | Two fixture records produce the expected CSV rows |
| 8 | — | Rent the GPU, run §6, commit the records **before terminating the pod** | — |
| 9 | `[Docs] Add batch-size sweep results` | Plots, tables, the committed records, and the predictions from §6 scored against what happened | — |

Commits 3, 4, and 5 are independent of each other — reorder freely. Commit 2
must precede 4, since MBU divides by `s_per_decode_step`.

If the day-per-branch rhythm is tight, 3+4 fold together cleanly (both are byte
arithmetic on the same model config) and so do 6+7.

---

## 6. What to run

Qwen3-8B / bf16 / greedy, **N=128**, unless stated.

| ID | Experiment | Configuration | What it answers |
|---|---|---|---|
| **E0** | Calibration | `--num-requests 4 --repeats 1`; check against the anchors below | Catches a broken timed region before you spend an hour |
| **E1** | Baseline ablation | `sequential`, then `static` at bs=1 | Separates "no longer re-copying the KV cache" from "actually batching". Re-run at session end as a drift check |
| **E2** | **Batch-size sweep** (headline) | bs ∈ {1,2,4,8,16,32,64,128} × three workloads | The curve, the elbow, where MBU saturates, where memory ends it |
| **E4** | Model-scale contrast | Qwen3-{0.6B, 8B} × bs ∈ {1,8,32}, `uniform-512x128` | Flat + 2% MBU at 0.6B = host-bound; rising MBU at 8B = GPU-bound |

Capacity (peak memory vs bs, and where it OOMs) falls out of E2 for free.

**Sanity anchors — memorize these before renting.** bs=1 decode ceiling on H100
SXM at 3.35 TB/s: Qwen3-0.6B **2815 tok/s**, Qwen3-8B **205 tok/s**. Exceeding
one means the timed region isn't covering the work you think it is. Landing far
below means host overhead owns the difference — expected at 0.6B, a bug at 8B
and high bs.

**Three plots.** (1) throughput vs batch size, one line per workload;
(2) `s_per_decode_step` vs batch size — the elbow; (3) peak memory and
`kv_efficiency` vs batch size, OOM points marked. Add MBU vs batch size with the
100% roofline drawn in.

**Write the predictions down before running.** Throughput near-linear at small bs
(step time barely moves — host-bound), bending as bs grows, plateauing once
decode is genuinely bandwidth-bound. Step time flat, then rising linearly, elbow
at the same bs. `sequential` → `static` bs=1 should be a *large* gap, because the
naive cache's `torch.cat` (`naive.py:32-33`) re-copies the entire cache on every
one of the 128 decode steps. Scoring these afterward is what turns a surprise
into a finding.

---

## 7. Exit criteria, and what comes next

Phase 1 is done when every E2 point has a committed record from a clean tree, the
elbow in `s_per_decode_step` has a location *and* a physical explanation, and you
can predict the memory ceiling from `weight_bytes + kv_bytes_reserved` before
running the point.

**Phase 2 — continuous batching.** Open with `--wave-order sorted` on static
batching: it tells you how much of static batching's loss was merely
*scheduling*, which is the honest framing of what continuous batching buys. Then
add `arrival_s`, `--request-rate` (Poisson), and the latency metrics §3 cut —
they stop being redundant the moment requests arrive over time, and the
throughput-vs-p99 plot becomes a real open-loop curve.

**Phase 3 — paged attention.** `kv_efficiency` and `kv_bytes_reserved` were
already measured in Phase 1; this is the phase that moves them. Re-run E2's
memory curve against the Phase 1 records.

**Phase 4 — prefix caching.** Add the `shared-prefix` workload. **Read
`PLAN.md` §4.10 first:** once cache state survives a request, the warm-up pass
leaves every prompt cached and repeats 2 and 3 hit repeat 1's cache — every timed
pass would report a 100% hit rate. Either reset between passes or declare "steady
state, warm cache" as the measured quantity and label the record. Decide it
deliberately; do not inherit it by accident.
