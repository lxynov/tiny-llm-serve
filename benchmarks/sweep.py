"""Batch-size sweep driver.

Walks a workload x batch-size grid, running each trial as a separate
`bench_throughput` process and leaving the records behind. Three properties
matter more than anything else this file does:

- **A fresh process per trial.** The caching allocator's pool survives within a
  process, so trials sharing one would report each other's peak memory -- the
  metric that decides where the sweep ends.
- **A failed trial does not end the sweep.** Running out of memory at a large
  batch size is a *result*: it is where capacity ran out. Aborting there would
  also discard every trial still queued behind it.
- **Resumable.** Trials already recorded are skipped, so an interrupted sweep
  can be restarted without re-paying for what it has.

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
class Conditions:
    """What a sweep holds fixed across every trial in it."""

    model: str
    dtype: str
    num_requests: int
    seed: int


@dataclass(frozen=True)
class Trial:
    workload: str
    mode: str
    batch_size: int | None


TrialId = tuple[Conditions, Trial]


def trial_id(conditions: Conditions, trial: Trial) -> TrialId:
    return (conditions, trial)


def recorded_trials(output_dir: Path) -> set[TrialId]:
    """Ids of the trials `output_dir` already holds records for."""
    ids = set()
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            ids.add(
                trial_id(
                    Conditions(
                        record["model"],
                        record["config"]["dtype"],
                        record["num_requests"],
                        record["seed"],
                    ),
                    Trial(
                        record["workload"],
                        record["engine_mode"],
                        record["config"]["batch_size"],
                    ),
                )
            )
        except (OSError, ValueError, KeyError):
            continue  # not one of ours, or written by a run that died
    return ids


def plan(args: argparse.Namespace) -> list[Trial]:
    """The grid, workload by workload, batch sizes ascending."""
    trials = []
    for workload in args.workloads:
        if args.sequential:
            trials.append(Trial(workload, "sequential", None))
        trials += [Trial(workload, "static", size) for size in sorted(args.batch_sizes)]
    return trials


def label(trial: Trial) -> str:
    name = f"{trial.workload} {trial.mode}"
    return name if trial.batch_size is None else f"{name} bs={trial.batch_size}"


def exit_code(ran: int, failed: int) -> int:
    return 1 if failed and ran == 0 else 0


def run(args: argparse.Namespace, trial: Trial) -> bool:
    command = [
        sys.executable,
        "-m",
        "benchmarks.bench_throughput",
        "--model",
        args.model,
        "--workload",
        trial.workload,
        "--mode",
        trial.mode,
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
    if trial.batch_size is not None:
        command += ["--batch-size", str(trial.batch_size)]
    if args.device is not None:
        command += ["--device", args.device]
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
    for size in args.batch_sizes:
        check_whole_waves(args.num_requests, size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    recorded = recorded_trials(args.output_dir)
    fixed = Conditions(args.model, args.dtype, args.num_requests, args.seed)
    trials = plan(args)
    skipped, failed = [], []
    for index, trial in enumerate(trials, start=1):
        progress = f"[{index}/{len(trials)}] {label(trial)}"
        if trial_id(fixed, trial) in recorded:
            print(f"{progress}: already recorded, skipping")
            skipped.append(trial)
            continue
        print(progress)
        if not run(args, trial):
            print(f"  FAILED: {label(trial)}", file=sys.stderr)
            failed.append(trial)

    ran = len(trials) - len(skipped) - len(failed)
    print(f"\n{ran} run, {len(skipped)} skipped, {len(failed)} failed")
    for trial in failed:
        print(f"  failed: {label(trial)}")
    return exit_code(ran, len(failed))


if __name__ == "__main__":
    sys.exit(main())
