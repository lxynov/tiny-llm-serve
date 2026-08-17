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
- **The ceiling is found once.** A batch size that ran out of memory records
  that fact, and every larger size on the same workload is then taken as
  hopeless rather than attempted -- memory demand only rises with the batch
  size, so the rest of the ladder would buy a model load apiece to confirm
  what the first failure already established. Other workloads keep going.
- **Resumable.** Trials already recorded are skipped, so an interrupted sweep
  can be restarted without re-paying for what it has -- including the
  out-of-memory records, which put the ceiling back before the restart can
  climb into it again.

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

from benchmarks.bench_throughput import (
    DTYPES,
    OOM_EXIT,
    RESULTS_DIR,
    check_whole_waves,
)
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


def recorded_trials(output_dir: Path) -> dict[TrialId, str]:
    """The trials `output_dir` already holds records for, each mapped to how it
    ended.

    How it ended matters as much as that it ran: a resumed sweep has to
    rediscover the ceiling a previous one found, or it walks straight back into
    it at the next batch size up. Records written before the field existed
    finished, so they read as "ok".
    """
    ids = {}
    for path in sorted(output_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            ids[
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
            ] = record.get("status", "ok")
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
    """Nonzero only when the sweep itself is broken. Trials that ran out of
    memory measured their ceiling, so they count as neither."""
    return 1 if failed and ran == 0 else 0


def note_ceiling(ceiling: dict[str, int], trial: Trial) -> None:
    """Remember the smallest batch size that did not fit on this workload."""
    if trial.batch_size is None:
        return  # the sequential baseline is not on the batch-size ladder
    limit = ceiling.get(trial.workload, trial.batch_size)
    ceiling[trial.workload] = min(limit, trial.batch_size)


def over_ceiling(ceiling: dict[str, int], trial: Trial) -> bool:
    """Whether a batch size this workload has already failed to fit is at or
    below this one.

    Memory demand rises with the batch size, so every size above one that ran
    out is a foregone conclusion -- and confirming it costs a model load apiece
    to learn nothing. The assumption is only safe within a workload: sizes are
    compared against the ceiling of their own curve, never against another
    workload's.
    """
    limit = ceiling.get(trial.workload)
    return (
        limit is not None and trial.batch_size is not None and trial.batch_size >= limit
    )


def run(args: argparse.Namespace, trial: Trial) -> str:
    """Run one trial in its own process: "ok", "oom", or "failed"."""
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
    code = subprocess.run(command).returncode
    if code == 0:
        return "ok"
    # Only a trial that recognized its own allocation failure and recorded it
    # exits this way. Any other nonzero code is a crash, and reading a crash as
    # a capacity ceiling would silently truncate the curve.
    return "oom" if code == OOM_EXIT else "failed"


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
    skipped, oom, unfit, failed = [], [], [], []
    ceiling: dict[str, int] = {}
    for index, trial in enumerate(trials, start=1):
        progress = f"[{index}/{len(trials)}] {label(trial)}"
        status = recorded.get(trial_id(fixed, trial))
        if status is not None:
            print(f"{progress}: already recorded, skipping")
            skipped.append(trial)
            if status == "oom":
                note_ceiling(ceiling, trial)
            continue
        if over_ceiling(ceiling, trial):
            limit = ceiling[trial.workload]
            print(f"{progress}: not attempted, bs={limit} already ran out of memory")
            unfit.append(trial)
            continue
        print(progress)
        outcome = run(args, trial)
        if outcome == "oom":
            print(f"  OUT OF MEMORY: {label(trial)}; the ceiling for this workload")
            note_ceiling(ceiling, trial)
            oom.append(trial)
        elif outcome == "failed":
            print(f"  FAILED: {label(trial)}", file=sys.stderr)
            failed.append(trial)

    ran = len(trials) - len(skipped) - len(oom) - len(unfit) - len(failed)
    print(
        f"\n{ran} run, {len(skipped)} skipped, {len(oom)} out of memory, "
        f"{len(unfit)} not attempted, {len(failed)} failed"
    )
    for trial in oom:
        print(f"  out of memory: {label(trial)}")
    for trial in unfit:
        print(f"  not attempted: {label(trial)}")
    for trial in failed:
        print(f"  failed: {label(trial)}")
    return exit_code(ran, len(failed))


if __name__ == "__main__":
    sys.exit(main())
