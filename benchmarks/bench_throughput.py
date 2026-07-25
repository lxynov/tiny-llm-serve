"""Offline throughput benchmark.

Runs a named workload through the engine and writes a JSON record to
benchmarks/results/, following the protocol in the README: pre-tokenized
prompts, ignore_eos so output lengths are exact, and the wall time of one
full pass over the workload.

Usage (from the repo root):
    python -m benchmarks.bench_throughput --model Qwen/Qwen3-0.6B \
        --workload mixed-out --num-requests 8
"""

import argparse
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.workloads import WORKLOADS, Request, build_workload
from tiny_llm_serve.engine.llm_engine import LLM
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.models import loader

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
RESULTS_DIR = Path(__file__).parent / "results"


def run_sequential(llm: LLM, requests: list[Request]) -> list[list[int]]:
    """The batch=1 floor: one request at a time over the naive KV cache."""
    outputs = []
    for request in requests:
        params = SamplingParams(
            temperature=0.0, max_tokens=request.output_len, ignore_eos=True
        )
        outputs.append(llm.generate_ids(request.prompt_ids, params))
    return outputs


def git_state() -> tuple[str | None, bool | None]:
    """The commit and whether the tree was dirty.

    A commit alone does not identify the code that ran: two runs from the
    same commit with different uncommitted edits are indistinguishable
    otherwise, which is exactly the claim a record is supposed to settle.
    """
    git = ["git", "-C", str(Path(__file__).parent)]
    try:
        commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True)
        status = subprocess.check_output(git + ["status", "--porcelain"], text=True)
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
    """The machine and software stack the numbers came off.

    Comparability is a protocol rule, but a rule nobody can check after the
    fact is a hope; recording the environment turns "same conditions" into
    something a later reader can verify against another record. On MPS the
    accelerator is the CPU's own chip, so `cpu` identifies both.
    """
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
    if device == "cuda":
        properties = torch.cuda.get_device_properties()
        info |= {
            "gpu": properties.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_bytes": properties.total_memory,
            "cuda": torch.version.cuda,
        }
    return info


def bench(args: argparse.Namespace) -> dict:
    device = loader.resolve_device(args.device)
    llm = LLM(args.model, device=device, dtype=DTYPES[args.dtype])
    requests = build_workload(
        args.workload, args.num_requests, llm.model.config.vocab_size, args.seed
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    outputs = run_sequential(llm, requests)
    wall = time.perf_counter() - start

    prompt_tokens = sum(len(r.prompt_ids) for r in requests)
    output_tokens = sum(len(o) for o in outputs)
    commit, dirty = git_state()
    now = datetime.now(timezone.utc)
    return {
        # The device is in the name because mixing records from different
        # machines is the easiest way to break rule 1 by accident, and a
        # directory listing is where that gets caught. The exact chip lives in
        # `environment`; to keep whole machines apart, give each its own
        # --output-dir rather than lengthening the name.
        "run_id": f"{now:%Y%m%d-%H%M%S}-{device}-{args.mode}-{args.workload}"
        f"-{uuid.uuid4().hex[:6]}",
        "date": now.isoformat(timespec="seconds"),
        "commit": commit,
        "dirty": dirty,
        "model": args.model,
        "device": device,
        "environment": environment(device),
        "engine_mode": args.mode,
        "workload": args.workload,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "config": {"dtype": args.dtype},
        "metrics": {
            "wall_time_s": wall,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_tok_s": output_tokens / wall,
            "total_tok_s": (prompt_tokens + output_tokens) / wall,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
            # Sequential mode runs one sequence at a time, and the naive cache
            # reserves exactly what it stores, so KV efficiency is 1 by design.
            "peak_concurrent_seqs": 1,
            "kv_efficiency": 1.0,
        },
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["sequential"], default="sequential")
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    record = bench(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["metrics"], indent=2))
    print(f"wrote {path}")
    return record


if __name__ == "__main__":
    main()
