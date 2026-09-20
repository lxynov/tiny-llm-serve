"""Offline throughput benchmark.

Runs a named workload through the engine and writes a JSON record to
benchmarks/results/throughput/, following the protocol in the README: pre-tokenized
prompts, ignore_eos so output lengths are exact, and the wall time of one
full pass over the workload.

A trial that runs out of memory still writes a record -- `"status": "oom"`
with no metrics -- because the batch size that did not fit is the sweep's
capacity ceiling, which is a finding rather than a lost run.

Usage (from the repo root):
    python -m benchmarks.throughput.bench --model Qwen/Qwen3-0.6B \
        --workload mixed-out --num-requests 8
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from benchmarks import hardware
from benchmarks.records import DTYPES, OOM_EXIT, RESULTS_DIR, provenance
from benchmarks.roofline import decode_bytes_read, kv_bytes_per_token, weight_bytes
from benchmarks.throughput.workloads import WORKLOADS, Request, build_workload
from tiny_llm_serve.engine.llm_engine import LLM, Timing
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.models import loader

# A trial's records land beside the other trials of its sweep, in a folder of
# this benchmark's own.
OUTPUT_DIR = RESULTS_DIR / "throughput"


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
    """The shared provenance block, plus what this benchmark was pointed at."""
    return provenance(args.model, device, args.mode, args.workload) | {
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
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
