"""Offline throughput benchmark.

Runs a named workload through the engine and writes a JSON record to
benchmarks/results/, following the protocol in the README: pre-tokenized
prompts, ignore_eos so output lengths are exact, and the wall time of one
full pass over the workload.

A trial that runs out of memory still writes a record -- `"status": "oom"`
with no metrics -- because the batch size that did not fit is the sweep's
capacity ceiling, which is a finding rather than a lost run.

Usage (from the repo root):
    python -m benchmarks.bench_throughput --model Qwen/Qwen3-0.6B \
        --workload mixed-out --num-requests 8
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks import hardware
from benchmarks.workloads import WORKLOADS, Request, build_workload
from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.engine.llm_engine import LLM, Timing
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.models import loader

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
RESULTS_DIR = Path(__file__).parent / "results"
# Exit code for a run that recorded its own out-of-memory ceiling; see __main__.
OOM_EXIT = 2


def run_sequential(
    llm: LLM, requests: list[Request], timing: Timing | None = None
) -> list[list[int]]:
    """The batch=1 floor: one request at a time over the naive KV cache."""
    outputs = []
    for request in requests:
        params = SamplingParams(
            temperature=0.0, max_tokens=request.output_len, ignore_eos=True
        )
        outputs.append(llm.generate_ids(request.prompt_ids, params, timing=timing))
    return outputs


def waves(requests: list[Request], batch_size: int) -> list[list[Request]]:
    return [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]


def check_whole_waves(num_requests: int, batch_size: int) -> None:
    """Reject a request count that leaves a short final wave."""
    if num_requests % batch_size:
        raise ValueError(
            f"--num-requests {num_requests} is not a multiple of --batch-size "
            f"{batch_size}: the last wave would run "
            f"{num_requests % batch_size} sequences but be recorded as "
            f"{batch_size}. Use a request count divisible by the batch size."
        )


def wave_model_len(wave: list[Request]) -> int:
    return max(len(r.prompt_ids) for r in wave) + max(r.output_len for r in wave)


def run_static(
    llm: LLM, requests: list[Request], batch_size: int, timing: Timing | None = None
) -> list[list[int]]:
    """Static batching over preallocated KV slots, one wave at a time."""
    outputs = []
    for wave in waves(requests, batch_size):
        params = [
            SamplingParams(temperature=0.0, max_tokens=r.output_len, ignore_eos=True)
            for r in wave
        ]
        outputs += llm.generate_batch_ids(
            [r.prompt_ids for r in wave],
            params,
            max_model_len=wave_model_len(wave),
            timing=timing,
        )
    return outputs


def weight_bytes(model: torch.nn.Module) -> int:
    """Bytes of parameters the device holds."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def kv_bytes_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    """Bytes one cached token occupies across every layer."""
    return (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * dtype.itemsize
    )


def decode_groups(
    requests: list[Request], mode: str, batch_size: int
) -> list[tuple[int, int, int]]:
    """Per set of sequences decoded together: (sequences, the KV length they
    start from, decode steps).
    """
    if mode == "static":
        return [
            (
                len(wave),
                max(len(r.prompt_ids) for r in wave),
                # The wave's first token falls out of the prefill logits.
                max(r.output_len for r in wave) - 1,
            )
            for wave in waves(requests, batch_size)
        ]
    return [(1, len(r.prompt_ids), r.output_len) for r in requests]


def decode_bytes_read(
    groups: list[tuple[int, int, int]], weights: int, kv_per_token: int
) -> int:
    """Bytes decode has to move, at minimum, to produce a pass's tokens.

    Every step re-reads all the weights to advance each sequence by one token,
    so the weight term is charged once per *step* rather than once per
    sequence. That is the batching win stated in bytes: it is why this total
    rises far more slowly than the batch size does, and why utilization climbs
    with it. On top of it each sequence reads its own KV window, which grows by
    one token every step.
    """
    total = 0
    for batch, kv_len, steps in groups:
        total += steps * weights
        window_tokens = steps * kv_len + steps * (steps + 1) // 2
        total += batch * window_tokens * kv_per_token
    return total


def length_stats(lengths: list[int]) -> dict:
    return {
        "mean": statistics.fmean(lengths),
        "median": statistics.median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }


def workload_stats(requests: list[Request]) -> dict:
    return {
        "prompt_len": length_stats([len(r.prompt_ids) for r in requests]),
        "output_len": length_stats([r.output_len for r in requests]),
    }


def git_state(repo: Path = Path(__file__).parent) -> tuple[str | None, bool | None]:
    """The commit and whether the tree carried uncommitted changes to it.

    A commit alone does not identify the code that ran: two runs from the
    same commit with different uncommitted edits are indistinguishable
    otherwise, which is exactly the claim a record is supposed to settle.

    Untracked files do not count, because this harness writes its records
    *into* the repository: counting them would mark every run after the first
    dirty for the file its predecessor left behind, and a flag that fires on
    every run says nothing about any of them.
    """
    git = ["git", "-C", str(repo)]
    try:
        commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True)
        status = subprocess.check_output(
            git + ["status", "--porcelain", "--untracked-files=no"], text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit.strip(), bool(status.strip())


def cpu_name() -> str | None:
    """The chip model, which `platform.processor()` is uselessly vague about
    ("arm" on macOS, "x86_64" on Linux) while it dominates any CPU run."""
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or None


def environment(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        # Torch sizes its thread pool from the machine, not from anything this
        # harness passes it, and CPU throughput scales with it.
        "torch_threads": torch.get_num_threads(),
    }
    if loader.is_cuda(device):
        properties = torch.cuda.get_device_properties(device)
        info |= {
            "gpu": properties.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_bytes": properties.total_memory,
            "cuda": torch.version.cuda,
        }
    return info


def run_config(args: argparse.Namespace) -> dict:
    """What the harness was told to do."""
    return {
        "dtype": args.dtype,
        "batch_size": args.batch_size if args.mode == "static" else None,
        "num_waves": (
            args.num_requests // args.batch_size if args.mode == "static" else None
        ),
    }


def identity(args: argparse.Namespace, device: str) -> dict:
    commit, dirty = git_state()
    now = datetime.now(timezone.utc)
    return {
        "run_id": f"{now:%Y%m%d-%H%M%S}-{device}-{args.mode}-{args.workload}"
        f"-{uuid.uuid4().hex[:6]}",
        "date": now.isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model": args.model,
        "device": device,
        "environment": environment(device),
        "engine_mode": args.mode,
        "load": "offline-drain",
        "workload": args.workload,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "config": run_config(args),
    }


def bench(args: argparse.Namespace) -> dict:
    """One trial's record: what it measured, or the ceiling it found."""
    if args.mode == "static":
        check_whole_waves(args.num_requests, args.batch_size)
    device = loader.resolve_device(args.device)
    try:
        return measure(args, device)
    except torch.OutOfMemoryError as error:
        return identity(args, device) | {
            "status": "oom",
            "error": str(error).splitlines()[0],
            "metrics": {},
        }


def measure(args: argparse.Namespace, device: str) -> dict:
    """Run the timed pass and describe what it moved."""
    on_gpu = loader.is_cuda(device)
    llm = LLM(args.model, device=device, dtype=DTYPES[args.dtype])
    requests = build_workload(
        args.workload, args.num_requests, llm.model.config.vocab_size, args.seed
    )

    def runner(timing: Timing | None = None) -> list[list[int]]:
        if args.mode == "static":
            return run_static(llm, requests, args.batch_size, timing)
        return run_sequential(llm, requests, timing)

    if on_gpu:
        # Per device: the stats these clear and the ones read below both belong
        # to whichever GPU they are pointed at, not to whichever is current.
        torch.cuda.reset_peak_memory_stats(device)
    # One Timing for the pass: the generate calls accumulate into it, so it
    # ends up holding the pass's totals over every request or wave.
    timing = Timing()
    loader.synchronize(device)
    start = time.perf_counter()
    outputs = runner(timing)
    loader.synchronize(device)
    wall = time.perf_counter() - start

    prompt_tokens = sum(len(r.prompt_ids) for r in requests)
    output_tokens = sum(len(o) for o in outputs)
    used_kv_tokens = prompt_tokens + output_tokens
    if args.mode == "static":
        reserved_kv_tokens = sum(
            len(w) * wave_model_len(w) for w in waves(requests, args.batch_size)
        )
        peak_concurrent_seqs = max(len(w) for w in waves(requests, args.batch_size))
    else:
        # The naive sequential cache grows exactly with what it stores.
        reserved_kv_tokens = used_kv_tokens
        peak_concurrent_seqs = 1
    weights = weight_bytes(llm.model)
    kv_per_token = kv_bytes_per_token(
        llm.model.config, next(llm.model.parameters()).dtype
    )
    decode_bytes = decode_bytes_read(
        decode_groups(requests, args.mode, args.batch_size), weights, kv_per_token
    )
    record = identity(args, device)
    peak_bytes_s = hardware.peak_hbm_bytes_s(record["environment"].get("gpu"))
    decode_time = timing.decode_s
    decode_steps = timing.decode_steps
    return record | {
        "workload_stats": workload_stats(requests),
        "status": "ok",
        "metrics": {
            "wall_time_s": wall,
            "prefill_time_s": timing.prefill_s,
            "decode_time_s": decode_time,
            "decode_steps": decode_steps,
            "s_per_decode_step": decode_time / decode_steps if decode_steps else None,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_tok_s": output_tokens / wall,
            "total_tok_s": (prompt_tokens + output_tokens) / wall,
            "decode_bytes_read": decode_bytes,
            "mbu": (
                decode_bytes / decode_time / peak_bytes_s
                if peak_bytes_s and decode_time
                else None
            ),
            "weight_bytes": weights,
            "kv_bytes_reserved": reserved_kv_tokens * kv_per_token,
            "peak_gpu_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if on_gpu else None
            ),
            "peak_gpu_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if on_gpu else None
            ),
            "peak_concurrent_seqs": peak_concurrent_seqs,
            "kv_efficiency": used_kv_tokens / reserved_kv_tokens,
        },
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["sequential", "static"], default="sequential"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="static mode only")
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    record = bench(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] == "oom":
        print(f"out of memory: {record['error']}", file=sys.stderr)
    else:
        print(json.dumps(record["metrics"], indent=2))
    print(f"wrote {path}")
    return record


if __name__ == "__main__":
    # A recorded ceiling is neither a success nor a crash, and the sweep driver
    # tells the three apart by exit code: it stops raising the batch size for
    # this workload on OOM, and keeps going on anything else.
    raise SystemExit(OOM_EXIT if main()["status"] == "oom" else 0)
