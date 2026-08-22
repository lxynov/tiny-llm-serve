"""Profile one decode step.

[`docs/raising-decode-mbu.md`](../docs/raising-decode-mbu.md) attributes decode's
18-25% MBU to two unrelated diseases and asks for a trace to tell them apart.
This script runs the three measurements that decides between them, on one wave
of one workload:

  1. which SDPA backend decode's call is eligible for, and which one it runs
     (section 5) -- asked of the backends directly, then confirmed in the trace;
  2. what the step copies -- the KV gather, and any expansion the chosen
     backend adds on top of it (section 2);
  3. whether the host or the device paces the step (section 4), by timing the
     step twice: once to the end of dispatch, once to the end of the device
     work it queued.

It drives prefill and the decode steps itself rather than calling
`LLM.generate_batch_ids`, so the profiler's window holds decode steps and
nothing else. The step mirrors the engine's (`engine/llm_engine.py:173-197`):
same model call, same KV manager, same sampler. What it leaves out is the
engine's per-row finish bookkeeping, which is the host-side sync section 3 is
about; `--engine-syncs` puts it back, so a pair of runs prices it.

Usage (from the repo root):
    python -m benchmarks.profile_decode --model Qwen/Qwen3-8B --dtype bfloat16 \
        --batch-size 64 --steps 8
"""

import argparse
import contextlib
import statistics
import time
import warnings
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks import hardware
from benchmarks.bench_throughput import DTYPES, kv_bytes_per_token, weight_bytes
from benchmarks.workloads import WORKLOADS, build_workload
from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.engine.llm_engine import LLM
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.kv import PreallocatedKVManager
from tiny_llm_serve.models import loader

TRACES_DIR = Path(__file__).parent / "traces"

# The aten op each SDPA backend lowers to, so the trace can be read for which
# one ran. `F.scaled_dot_product_attention` is a composite: the profiler records
# the dispatch above and the chosen kernel below it.
SDPA_OPS = {
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "flash-cpu",
    "aten::_scaled_dot_product_efficient_attention": "mem-efficient",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_attention_math": "math",
}

# Ops the doc predicts, and what finding one in the step would mean. Every one
# of them is a copy the modelled byte count does not charge for, so their share
# of device time is the gap between reported and real MBU.
WATCHED_OPS = {
    "aten::index": "KV gather: advanced indexing copies the window (section 2)",
    "aten::index_put_": "KV write: this step's token into its slot",
    "aten::repeat_interleave": "GQA expansion: k/v materialized at num_heads",
    "aten::contiguous": "a view being materialized",
    "aten::clone": "a copy under another name",
    "aten::copy_": "any remaining copy, including dtype casts",
    "aten::_softmax": "attention scores materialized (math backend)",
}


def sdpa_context(name: str):
    """Pin SDPA to one backend, or leave the choice to torch.

    Pinning is the experiment the doc's section 5 ends on: the default priority
    order is flash, mem-efficient, math, cudnn, and math is always eligible, so
    cudnn is unreachable by default however well it would do here.
    """
    if name == "auto":
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    return sdpa_kernel([backends[name]])


def sdpa_eligibility(
    config: ModelConfig, batch: int, kv_len: int, device: str, dtype: torch.dtype
) -> list[tuple[str, bool, list[str]]]:
    """Ask each fused backend whether it would take decode's exact call.

    Cheaper and more direct than reading it out of a trace: the same C++ checks
    the dispatcher runs, with `debug=True` so a refusal names its reason. The
    call described is the one `layers/attention.py` makes at decode -- a bool
    mask from `kv/preallocated.py`, `enable_gqa=True`, one query position.
    """
    if not loader.is_cuda(device):
        return []
    from torch.backends.cuda import (
        SDPAParams,
        can_use_cudnn_attention,
        can_use_efficient_attention,
        can_use_flash_attention,
    )

    query = torch.empty(
        batch,
        config.num_attention_heads,
        1,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.empty(
        batch,
        config.num_key_value_heads,
        kv_len,
        config.head_dim,
        device=device,
        dtype=dtype,
    )
    mask = torch.ones(batch, 1, 1, kv_len, dtype=torch.bool, device=device)
    params = SDPAParams(query, key, torch.empty_like(key), mask, 0.0, False, True)
    report = []
    for name, can_use in (
        ("flash", can_use_flash_attention),
        ("mem-efficient", can_use_efficient_attention),
        ("cudnn", can_use_cudnn_attention),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            usable = can_use(params, True)
        report.append((name, usable, [str(w.message).strip() for w in caught]))
    return report


class DecodeLoop:
    """One prefilled wave, stepped the way the engine steps it.

    Shapes:
        lens:        [batch] int64 -- each row's cached length, the position
                     its next token lands on
        step_logits: [batch, vocab_size] -- last step's output, this step's input
    """

    def __init__(
        self,
        llm: LLM,
        prompts: list[list[int]],
        max_model_len: int,
        engine_syncs: bool = False,
    ) -> None:
        self.llm = llm
        self.engine_syncs = engine_syncs
        self.params = SamplingParams(temperature=0.0, ignore_eos=True)
        prompt_lens = [len(p) for p in prompts]
        self.manager = PreallocatedKVManager(
            llm.model.config,
            num_slots=len(prompts),
            max_model_len=max_model_len,
            device=llm.device,
            dtype=next(llm.model.parameters()).dtype,
        )
        self.slots = [self.manager.admit(n) for n in prompt_lens]
        padded_len = max(prompt_lens)
        input_ids = torch.zeros((len(prompts), padded_len), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            input_ids[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        input_ids = input_ids.to(llm.device)
        positions = torch.arange(padded_len, device=llm.device).expand(len(prompts), -1)
        self.lens = torch.tensor(prompt_lens, device=llm.device)
        self.finished = torch.zeros(len(prompts), dtype=torch.bool, device=llm.device)
        self.step_logits = llm.model(
            input_ids,
            positions,
            self.manager.begin_prefill(self.slots, prompt_lens),
            self.lens - 1,
        )

    def step(self) -> None:
        """Sample one token per row and advance every sequence by it."""
        next_ids = self.llm.sampler(self.step_logits, self.params)
        if self.engine_syncs:
            # llm_engine.py:176-186, verbatim in cost if not in effect: one
            # device read per sequence, plus the two the loop ends on.
            for i, _ in enumerate(next_ids.tolist()):
                if self.finished[i]:
                    continue
            bool(self.finished.all())
        logits = self.llm.model(
            next_ids.unsqueeze(1),
            self.lens.unsqueeze(1),
            self.manager.begin_decode(self.slots),
        )
        self.step_logits = logits[:, 0]
        self.lens = self.lens + 1

    @property
    def kv_len(self) -> int:
        """Keys the next step will attend over -- the longest row, plus its own
        token."""
        return int(self.lens.max()) + 1


def time_steps(
    loop: DecodeLoop, count: int, device: str
) -> tuple[list[float], list[float]]:
    """Per-step (host seconds, wall seconds), one entry each per step.

    Section 4's measurement. The first clock read stops when the host has
    finished queuing the step, the second when the device has finished running
    it. If the two agree, the GPU emptied its queue before the host could
    refill it, and decode is host-bound: the fix is fewer dispatches (CUDA
    graphs, torch.compile), not fewer bytes.

    Every step is kept rather than averaged because decode steps are not
    interchangeable: kv_len grows by one each step, so a backend that plans per
    shape re-plans every step, and one that does not pays only for the first.
    A mean cannot tell those apart; the spread can.
    """
    host, wall = [], []
    for _ in range(count):
        loader.synchronize(device)
        start = time.perf_counter()
        loop.step()
        dispatched = time.perf_counter()
        loader.synchronize(device)
        host.append(dispatched - start)
        wall.append(time.perf_counter() - start)
    return host, wall


def profile_steps(
    loop: DecodeLoop, count: int, device: str, trace: Path | None
) -> profile:
    """Run `count` steps under the profiler, exporting a chrome trace."""
    activities = [ProfilerActivity.CPU]
    if loader.is_cuda(device):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(count):
            loop.step()
        loader.synchronize(device)
    if trace is not None:
        trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace))
    return prof


def self_device_us(event) -> float:
    """Self device time in microseconds, under either of torch's names for it."""
    for name in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, name, None)
        if value:
            return float(value)
    return 0.0


def per_step(prof: profile, steps: int) -> dict:
    """Reduce a trace to the counts and times the doc argues over."""
    events = prof.key_averages()
    aten = [e for e in events if e.key.startswith("aten::")]
    return {
        "steps": steps,
        # Nested ops count too (aten::linear contains aten::mm), which is the
        # same thing the doc counts when it estimates ~2,000 dispatches a step.
        "aten_calls_per_step": sum(e.count for e in aten) / steps,
        "self_device_ms_per_step": sum(self_device_us(e) for e in aten) / steps / 1e3,
        "self_cpu_ms_per_step": sum(e.self_cpu_time_total for e in aten) / steps / 1e3,
        "sdpa_backends": sorted({SDPA_OPS[e.key] for e in aten if e.key in SDPA_OPS}),
        "ops": {
            e.key: {
                "calls_per_step": e.count / steps,
                "device_ms_per_step": self_device_us(e) / steps / 1e3,
                "cpu_ms_per_step": e.self_cpu_time_total / steps / 1e3,
            }
            for e in aten
            if e.key in SDPA_OPS or e.key in WATCHED_OPS
        },
    }


def charged_bytes_per_step(model, batch: int, kv_len: int) -> int:
    """The bytes `bench_throughput.decode_bytes_read` would charge this step:
    the weights once, plus each row's KV window. Everything the trace shows on
    top of this is what MBU does not count."""
    kv_per_token = kv_bytes_per_token(model.config, next(model.parameters()).dtype)
    return weight_bytes(model) + batch * kv_len * kv_per_token


def report(summary: dict, prof: profile, rows: int) -> None:
    """Print the three answers, then the raw table behind them."""
    print("\n== SDPA eligibility (asked of the backends, before the trace) ==")
    if not summary["eligibility"]:
        print("  (CUDA only)")
    for name, usable, reasons in summary["eligibility"]:
        print(f"  {name:<14} {'usable' if usable else 'REFUSED'}")
        if not usable:
            for reason in reasons:
                print(f"      {reason}")

    print("\n== what ran ==")
    backends = summary["sdpa_backends"]
    print(f"  SDPA backend in the trace: {', '.join(backends) or 'none recorded'}")
    print(f"  aten calls per step:       {summary['aten_calls_per_step']:.0f}")
    print(f"  self device time per step: {summary['self_device_ms_per_step']:.2f} ms")
    for key, stats in sorted(
        summary["ops"].items(), key=lambda kv: -kv[1]["device_ms_per_step"]
    ):
        note = WATCHED_OPS.get(key) or f"SDPA backend: {SDPA_OPS[key]}"
        print(
            f"  {key:<48} {stats['calls_per_step']:>7.1f}/step "
            f"{stats['device_ms_per_step']:>8.2f} ms  {note}"
        )

    print("\n== host or device (section 4) ==")
    host, wall = summary["host_s_per_step"], summary["wall_s_per_step"]
    for name, times in (("dispatch only ", host), ("through device", wall)):
        print(
            f"  {name}: median {statistics.median(times) * 1e3:7.2f} ms/step"
            f"   min {min(times) * 1e3:7.2f}   max {max(times) * 1e3:7.2f}"
        )
    print(f"  host share:     {statistics.median(host) / statistics.median(wall):.0%}")
    print("  per step (ms):  " + " ".join(f"{t * 1e3:.1f}" for t in wall))

    print("\n== the metric, on this step ==")
    print(f"  charged bytes/step: {summary['charged_bytes_per_step'] / 1e9:.2f} GB")
    if summary["mbu"] is not None:
        print(f"  reported MBU:       {summary['mbu']:.3f}")
    print(f"  (batch {summary['batch_size']}, kv_len {summary['kv_len']})")

    print(f"\n== top {rows} ops by self device time ==")
    sort_by = (
        "self_device_time_total" if summary["sdpa_backends"] else "self_cpu_time_total"
    )
    try:
        print(prof.key_averages().table(sort_by=sort_by, row_limit=rows))
    except (KeyError, AssertionError):
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=rows))
    if summary["trace"]:
        print(f"wrote {summary['trace']}  (open in chrome://tracing or perfetto.dev)")


@torch.inference_mode()
def profile_decode(args: argparse.Namespace) -> tuple[dict, profile]:
    """Prefill a wave, warm it up, time it, then profile it."""
    device = loader.resolve_device(args.device)
    llm = LLM(args.model, device=device, dtype=DTYPES[args.dtype])
    requests = build_workload(
        args.workload, args.batch_size, llm.model.config.vocab_size, args.seed
    )
    prompts = [r.prompt_ids for r in requests]
    # Warmup, timing and profiling all step the same wave, and a slot that runs
    # out of reservation raises rather than wrapping, so reserve for all three.
    total_steps = args.warmup + 2 * args.steps
    loop = DecodeLoop(
        llm,
        prompts,
        max_model_len=max(len(p) for p in prompts) + total_steps + 1,
        engine_syncs=args.engine_syncs,
    )
    with sdpa_context(args.sdpa_backend):
        for _ in range(args.warmup):
            loop.step()
        host_s, wall_s = time_steps(loop, args.steps, device)
        kv_len = loop.kv_len
        prof = profile_steps(loop, args.steps, device, args.trace)

    charged = charged_bytes_per_step(llm.model, len(prompts), kv_len)
    peak_bytes_s = hardware.peak_hbm_bytes_s(
        torch.cuda.get_device_properties(device).name
        if loader.is_cuda(device)
        else None
    )
    summary = per_step(prof, args.steps) | {
        "model": args.model,
        "device": device,
        "dtype": args.dtype,
        "workload": args.workload,
        "batch_size": len(prompts),
        "kv_len": kv_len,
        "sdpa_requested": args.sdpa_backend,
        "engine_syncs": args.engine_syncs,
        "eligibility": sdpa_eligibility(
            llm.model.config, len(prompts), kv_len, device, DTYPES[args.dtype]
        ),
        "host_s_per_step": host_s,
        "wall_s_per_step": wall_s,
        "charged_bytes_per_step": charged,
        # Off the median step, so one shape-planning outlier does not set it.
        "mbu": (
            charged / statistics.median(wall_s) / peak_bytes_s if peak_bytes_s else None
        ),
        "trace": str(args.trace) if args.trace else None,
    }
    return summary, prof


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--workload", choices=sorted(WORKLOADS), default="uniform-512x128"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--steps", type=int, default=8, help="steps timed, then profiled"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument(
        "--sdpa-backend",
        choices=["auto", "flash", "efficient", "cudnn", "math"],
        default="auto",
        help="pin SDPA to one backend; it raises if that backend refuses the call",
    )
    parser.add_argument(
        "--engine-syncs",
        action="store_true",
        help="add the engine's per-row finish bookkeeping back into the step",
    )
    parser.add_argument("--trace", type=Path, default=None, help="chrome trace path")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--rows", type=int, default=15, help="rows of the op table")
    args = parser.parse_args(argv)
    if args.trace is None and not args.no_trace:
        args.trace = TRACES_DIR / (
            f"decode-{Path(args.model).name}-{args.dtype}-bs{args.batch_size}"
            f"-{args.sdpa_backend}.json"
        )
    if args.no_trace:
        args.trace = None

    summary, prof = profile_decode(args)
    report(summary, prof, args.rows)
    return summary


if __name__ == "__main__":
    main()
