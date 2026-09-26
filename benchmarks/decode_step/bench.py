"""Decode-step microbenchmark.

Usage (from the repo root):
    python -m benchmarks.decode_step.bench --model Qwen/Qwen3-8B \
        --cells 1x576,8x576,64x576,256x576,64x2048 \
        --warmup 10 --steps 30 --repeats 3
"""

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks import hardware
from benchmarks.records import (
    DTYPES,
    OOM_EXIT,
    RESULTS_DIR,
    code_version,
    provenance,
    record_hardware,
    slug,
)
from benchmarks.roofline import decode_bytes_read, kv_bytes_per_token, weight_bytes
from tiny_llm_serve.engine.llm_engine import LLM, DecodeState
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.models import loader

OUTPUT_DIR = RESULTS_DIR / "decode-step"
# A cell's timed steps center on its kv_len. 576 is the mean cached length of
# `uniform-512x128` over its run (512 prompt + 128 output, halfway), so these
# land next to the baseline sweep.
DEFAULT_CELLS = "1x576,8x576,64x576,256x576,64x2048"


@dataclass(frozen=True)
class Cell:
    """One point to measure: a batch size and the KV length its timed steps
    center on."""

    batch: int
    kv_len: int

    def __str__(self) -> str:
        return f"{self.batch}x{self.kv_len}"


def parse_cells(text: str) -> list[Cell]:
    """`"1x576,64x2048"` -> the cells it names."""
    cells = []
    for part in text.split(","):
        batch, _, kv_len = part.strip().partition("x")
        try:
            cell = Cell(int(batch), int(kv_len))
        except ValueError:
            raise ValueError(f"bad cell {part!r}: expected BATCHxKV_LEN, e.g. 64x576")
        if cell.batch < 1 or cell.kv_len < 1:
            raise ValueError(f"bad cell {part!r}: batch and kv_len must be >= 1")
        cells.append(cell)
    if not cells:
        raise ValueError("no cells given")
    return cells


def synthetic_prompts(cell: Cell, vocab_size: int, seed: int) -> list[list[int]]:
    """`cell.batch` prompts of exactly `cell.kv_len` random token ids."""
    rng = random.Random(seed)
    return [
        [rng.randrange(vocab_size) for _ in range(cell.kv_len)]
        for _ in range(cell.batch)
    ]


def build_state(llm: LLM, cell: Cell, steps: int, seed: int) -> DecodeState:
    """Put `cell.batch` sequences in flight with `cell.kv_len` tokens cached.

    Shapes:
        state.manager.k_cache[layer], .v_cache[layer]:
            [num_slots, max_model_len, num_kv_heads, head_dim]
    """
    prompts = synthetic_prompts(cell, llm.model.config.vocab_size, seed)
    state = llm.prefill_batch(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=steps + 2, ignore_eos=True),
        max_model_len=cell.kv_len + steps + 1,
    )
    # Inference mode because the pool was allocated inside it: torch refuses an
    # in-place write to an inference tensor from anywhere else.
    with torch.inference_mode():
        for cache in (state.manager.k_cache, state.manager.v_cache):
            for tensor in cache:
                tensor[:, cell.kv_len :].uniform_(-1.0, 1.0)
    return state


def middle_step(args: argparse.Namespace) -> int:
    """Which step of a cell, warmup included and counting from 1, is the middle
    timed one. Step k attends the prompt plus k tokens, so a cell prefills its
    kv_len minus this, and the middle timed step attends exactly kv_len."""
    return args.warmup + (args.repeats * args.steps + 1) // 2


def time_steps(llm: LLM, state: DecodeState, steps: int, device: str) -> float:
    """Run `steps` decode steps; return the wall seconds they took."""
    loader.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        llm.decode_step(state)
    loader.synchronize(device)
    return time.perf_counter() - start


def measure_cell(
    llm: LLM,
    cell: Cell,
    args: argparse.Namespace,
    device: str,
    peak_bytes_s: float | None,
) -> dict:
    """One cell: prefill, warm up, then `--repeats` timed windows of `--steps`."""
    on_gpu = loader.is_cuda(device)
    if on_gpu:
        # Per cell, so the peak reads as this cell's: cells share a process
        # here, unlike a sweep's trials, because reloading an 8B checkpoint per
        # cell would cost more than the measurement.
        torch.cuda.reset_peak_memory_stats(device)
    total_steps = args.warmup + args.repeats * args.steps
    # Prefilled short of kv_len, so the timed steps center on it rather than
    # all running past it.
    prompt = Cell(cell.batch, cell.kv_len - middle_step(args))
    state = build_state(llm, prompt, total_steps, args.seed)
    for _ in range(args.warmup):
        llm.decode_step(state)

    weights = weight_bytes(llm.model)
    kv_per_token = kv_bytes_per_token(
        llm.model.config, next(llm.model.parameters()).dtype
    )
    # Tracked on the host rather than read back from `state.lens`: every row
    # starts at the prompt's length and every step appends one token to all
    # of them.
    cached = prompt.kv_len + args.warmup
    runs = []
    for _ in range(args.repeats):
        if state.done:
            raise RuntimeError(f"cell {cell} finished mid-run: budget too small")
        wall = time_steps(llm, state, args.steps, device)
        runs.append(
            {
                "s": wall,
                "s_per_decode_step": wall / args.steps,
                # Step j of this window attends cached + j positions, which is
                # exactly what decode_bytes_read charges a decode group for.
                "decode_bytes_read": decode_bytes_read(
                    [(cell.batch, cached, args.steps)], weights, kv_per_token
                ),
                "kv_len_first_step": cached + 1,
                "kv_len_last_step": cached + args.steps,
            }
        )
        cached += args.steps

    # The median *window*, not a median per column: the rate and the bytes
    # then both describe one run that actually happened.
    median = sorted(runs, key=lambda r: r["s"])[len(runs) // 2]
    per_step = median["s_per_decode_step"]
    bytes_s = median["decode_bytes_read"] / median["s"]
    return {
        "cell": str(cell),
        "batch": cell.batch,
        "kv_len": cell.kv_len,
        "status": "ok",
        "metrics": {
            "s_per_decode_step": per_step,
            "s_per_decode_step_min": min(r["s_per_decode_step"] for r in runs),
            "s_per_decode_step_max": max(r["s_per_decode_step"] for r in runs),
            "decode_bytes_read": median["decode_bytes_read"],
            "bytes_per_decode_step": median["decode_bytes_read"] / args.steps,
            "bytes_s": bytes_s,
            "mbu": bytes_s / peak_bytes_s if peak_bytes_s else None,
            "kv_len_first_step": runs[0]["kv_len_first_step"],
            "kv_len_last_step": runs[-1]["kv_len_last_step"],
            "peak_gpu_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if on_gpu else None
            ),
        },
        "runs": runs,
    }


def identity(args: argparse.Namespace, device: str) -> dict:
    """The shared provenance block, plus what this benchmark was told to do.

    The record is named for the UTC time it started: its folder already says
    the day, commit, chip, model and dtype, and reruns there sort into order.
    """
    name = f"{datetime.now(timezone.utc):%H%M%S}"
    return provenance(args.model, device, name) | {
        "benchmark": "decode-step",
        "seed": args.seed,
        "config": {
            "dtype": args.dtype,
            "cells": [str(c) for c in args.cells],
            "warmup": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
        },
    }


def run_folder(record: dict) -> str:
    """The folder under `--output-dir` a run's record belongs in.

    Named the way a throughput sweep's is: the UTC day the run started and the
    commit it measured lead, so a listing sorts into a history of the engine,
    then the chip, model and dtype that decide whether two runs' steps can be
    read against each other. Runs sharing all five share the folder -- a
    microbenchmark is rerun to check its noise or to try other cells -- and a
    step that got faster between two commits lands in two folders, not one.
    The cells are not in it -- each one carries its own shape, so a cell is
    compared with the same cell of another run, not with the run -- and
    neither is the seed, which only picks the prompts' token ids, and a step's
    cost does not depend on which ids those are.
    """
    return "-".join(
        (
            f"{datetime.fromisoformat(record['date']):%Y%m%d}",
            code_version(record["commit"], record["dirty"]),
            record_hardware(record),
            slug(Path(record["model"]).name),
            slug(record["config"]["dtype"]),
        )
    )


def bench(args: argparse.Namespace) -> dict:
    """Every cell, in the order given, in one process against one model."""
    device = loader.resolve_device(args.device)
    llm = LLM(args.model, device=device, dtype=DTYPES[args.dtype])
    identifiers = identity(args, device)
    peak_bytes_s = hardware.peak_hbm_bytes_s(identifiers["environment"].get("gpu"))
    record = identifiers | {
        "weight_bytes": weight_bytes(llm.model),
        "kv_bytes_per_token": kv_bytes_per_token(
            llm.model.config, next(llm.model.parameters()).dtype
        ),
    }
    cells = []
    for cell in args.cells:
        print(f"cell {cell}", file=sys.stderr)
        try:
            cells.append(measure_cell(llm, cell, args, device, peak_bytes_s))
        except torch.OutOfMemoryError as error:
            print(f"  out of memory: {cell}", file=sys.stderr)
            cells.append(
                {
                    "cell": str(cell),
                    "batch": cell.batch,
                    "kv_len": cell.kv_len,
                    "status": "oom",
                    "error": str(error).splitlines()[0],
                    "metrics": {},
                }
            )
        if loader.is_cuda(device):
            # The next cell is a different shape against the same pool, so hand
            # this one's blocks back rather than fragmenting around them.
            torch.cuda.empty_cache()
    return record | {"cells": cells}


COLUMNS: list[tuple[str, str, str]] = [
    # header, key, format
    ("cell", "cell", ""),
    ("status", "status", ""),
    ("ms/step", "s_per_decode_step", ".2f"),
    ("spread", "spread", ".1%"),
    ("GB/s", "bytes_s", ",.0f"),
    ("mbu", "mbu", ".3f"),
    ("kv_1st", "kv_len_first_step", ""),
    ("peak_GB", "peak_gpu_memory_allocated_bytes", ".1f"),
]
# Metrics reported in a unit other than the one recorded.
SCALE = {
    "s_per_decode_step": 1e3,
    "bytes_s": 1e-9,
    "peak_gpu_memory_allocated_bytes": 2**-30,
}


def cells_table(cells: list[dict]) -> list[str]:
    """The cells as one right-justified table, in the order they ran."""
    body = []
    for cell in cells:
        metrics = cell["metrics"]
        fastest = metrics.get("s_per_decode_step_min")
        row = []
        for _, key, spec in COLUMNS:
            value = cell.get(key, metrics.get(key))
            if key == "spread" and fastest:
                value = metrics["s_per_decode_step_max"] / fastest - 1
            if value is None:
                row.append("-")
            elif isinstance(value, str):
                row.append(value)
            else:
                row.append(format(value * SCALE.get(key, 1), spec))
        body.append(row)
    headers = [h for h, _, _ in COLUMNS]
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    return [
        line,
        "-" * len(line),
        *("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in body),
    ]


def summarize(record: dict) -> str:
    """The table, under the conditions every row of it shares."""
    env = record["environment"]
    config = record["config"]
    out = [
        f"{record['model']} {config['dtype']} on "
        f"{env.get('gpu') or env.get('cpu')}, "
        f"torch {env.get('torch', '?')}, "
        f"commit {(record.get('commit') or '?')[:12]}"
        f"{' (dirty)' if record.get('dirty') else ''}",
        f"{config['warmup']} warmup, {config['repeats']} x {config['steps']} "
        f"timed steps per cell",
        "",
        *cells_table(record["cells"]),
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--cells",
        type=parse_cells,
        default=parse_cells(DEFAULT_CELLS),
        help=f"comma-separated BATCHxKV_LEN points (default: {DEFAULT_CELLS})",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="untimed steps per cell first"
    )
    parser.add_argument("--steps", type=int, default=30, help="timed steps per window")
    parser.add_argument("--repeats", type=int, default=3, help="timed windows per cell")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="parent of the run folder"
    )
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("--steps and --repeats must be >= 1, --warmup >= 0")
    short = [str(c) for c in args.cells if c.kv_len <= middle_step(args)]
    if short:
        parser.error(
            f"cells {', '.join(short)}: kv_len must exceed {middle_step(args)}, "
            "the steps up to the middle timed one"
        )

    record = bench(args)
    run_dir = args.output_dir / run_folder(record)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(summarize(record))
    print(f"wrote {path}")
    return record


if __name__ == "__main__":
    # A run whose every cell hit the memory ceiling measured nothing, and says
    # so the way a throughput trial does.
    record = main()
    raise SystemExit(
        OOM_EXIT if all(c["status"] == "oom" for c in record["cells"]) else 0
    )
