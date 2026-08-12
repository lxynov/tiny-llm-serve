"""Batch-size sweep driver.

Walks a workload x batch-size grid, running each point as a separate
`bench_throughput` process and leaving the records behind. Three properties
matter more than anything else this file does:

- **A fresh process per point.** The caching allocator's pool survives within a
  process, so points sharing one would report each other's peak memory -- the
  metric that decides where the sweep ends.
- **A failed point does not end the sweep.** Running out of memory at a large
  batch size is a *result*: it is where capacity ran out. Aborting there would
  also throw away every point still queued behind it.
- **Resumable.** Points already recorded are skipped, so an interrupted sweep
  can be restarted without paying for what it already has.

Usage (from the repo root):
    python -m benchmarks.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
        --num-requests 128 --batch-sizes 1,2,4,8,16,32,64,128
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks.bench_throughput import DTYPES, RESULTS_DIR, check_whole_waves
from benchmarks.workloads import WORKLOADS


@dataclass(frozen=True)
class Point:
    workload: str
    mode: str
    batch_size: int | None


def point_key(
    model: str, dtype: str, num_requests: int, seed: int, point: Point
) -> tuple:
    """What makes two runs the same measurement.

    Everything a record must match on for one to stand in for the other. The
    filename cannot serve: it carries a timestamp and a random suffix, so it
    identifies a run rather than the point that run measured.
    """
    return (
        model,
        dtype,
        num_requests,
        seed,
        point.workload,
        point.mode,
        point.batch_size,
    )


def recorded_keys(output_dir: Path) -> set[tuple]:
    """Keys of the points `output_dir` already holds records for."""
    keys = set()
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            keys.add(
                point_key(
                    record["model"],
                    record["config"]["dtype"],
                    record["num_requests"],
                    record["seed"],
                    Point(
                        record["workload"],
                        record["engine_mode"],
                        record["config"]["batch_size"],
                    ),
                )
            )
        except (OSError, ValueError, KeyError):
            continue  # not one of ours, or written by a run that died
    return keys


def plan(args: argparse.Namespace) -> list[Point]:
    """The grid, workload by workload, batch sizes ascending.

    Ascending because the small points are the ones that fit: ordering this way
    means a sweep that dies at its memory ceiling has already banked the rest of
    the curve. The sequential baseline leads each workload -- it is the floor
    the batched points are measured against.
    """
    points = []
    for workload in args.workloads:
        if args.sequential:
            points.append(Point(workload, "sequential", None))
        points += [Point(workload, "static", size) for size in sorted(args.batch_sizes)]
    return points


def exit_code(ran: int, failed: int) -> int:
    """A sweep that lost its largest batch sizes to memory found the ceiling
    and did its job; one where nothing ran at all is broken."""
    return 1 if failed and ran == 0 else 0


def run(args: argparse.Namespace, point: Point) -> bool:
    command = [
        sys.executable,
        "-m",
        "benchmarks.bench_throughput",
        "--model",
        args.model,
        "--workload",
        point.workload,
        "--mode",
        point.mode,
        "--num-requests",
        str(args.num_requests),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
    ]
    if point.batch_size is not None:
        command += ["--batch-size", str(point.batch_size)]
    if args.device is not None:
        command += ["--device", args.device]
    # Output is left to stream: a sweep is something you watch, and a point
    # that dies should say why where you can see it.
    return subprocess.run(command).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--workloads",
        type=lambda s: s.split(","),
        default=sorted(WORKLOADS),
        help="comma-separated workload names",
    )
    parser.add_argument(
        "--batch-sizes",
        type=lambda s: [int(part) for part in s.split(",")],
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="also run the sequential baseline for each workload",
    )
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    unknown = set(args.workloads) - set(WORKLOADS)
    if unknown:
        parser.error(f"unknown workloads {sorted(unknown)}; known: {sorted(WORKLOADS)}")
    # Up front, so a grid that cannot be run says so before the first model
    # load rather than after every batch size that happened to divide.
    for size in args.batch_sizes:
        check_whole_waves(args.num_requests, size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = recorded_keys(args.output_dir)
    points = plan(args)
    skipped, failed = [], []
    for index, point in enumerate(points, start=1):
        label = f"{point.workload} {point.mode}"
        if point.batch_size is not None:
            label += f" bs={point.batch_size}"
        if (
            point_key(args.model, args.dtype, args.num_requests, args.seed, point)
            in done
        ):
            print(f"[{index}/{len(points)}] {label}: already recorded, skipping")
            skipped.append(point)
            continue
        print(f"[{index}/{len(points)}] {label}")
        if not run(args, point):
            print(f"  FAILED: {label}", file=sys.stderr)
            failed.append(point)

    ran = len(points) - len(skipped) - len(failed)
    print(f"\n{ran} run, {len(skipped)} skipped, {len(failed)} failed")
    for point in failed:
        print(f"  failed: {point.workload} {point.mode} bs={point.batch_size}")
    return exit_code(ran, len(failed))


if __name__ == "__main__":
    sys.exit(main())
