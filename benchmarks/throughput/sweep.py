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
- **One folder per sweep.** Records land together in a directory under
  `--output-dir`, named after the conditions the sweep holds fixed, next to a
  `sweep.json` index of the grid. Naming it after the conditions rather than
  the clock is what keeps the previous point true: a folder stamped with the
  start time would be a new empty one on every restart, and the resume would
  find nothing to skip.

Usage (from the repo root):
    python -m benchmarks.throughput.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
        --num-requests 128 --batch-sizes 1,2,4,8,16,32,64,128
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.records import DTYPES, OOM_EXIT
from benchmarks.throughput.bench import OUTPUT_DIR, check_whole_waves
from benchmarks.throughput.workloads import WORKLOADS
from tiny_llm_serve.models import loader

# The folder's index, and the one file in it that is not a trial.
MANIFEST = "sweep.json"


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


def slug(text: str) -> str:
    """`text` as a lowercase path component: alphanumerics, dots and dashes.

    Dots survive because model names carry them -- `qwen3-0.6b` reads as the
    checkpoint it names and `qwen3-0-6b` does not.
    """
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")


def sweep_id(conditions: Conditions, device: str) -> str:
    """The folder this sweep's records belong in.

    Named after the conditions rather than the clock, because the folder is
    what a restart reads: one stamped with the start time would be a new empty
    directory every run, and the sweep would re-buy a grid it already owns.
    Everything that decides whether two trials can be plotted against each
    other is in the name instead, so changing any of them writes elsewhere --
    the device included, which the record format otherwise asks you to keep
    apart by hand.
    """
    return "-".join(
        (
            slug(Path(conditions.model).name),
            slug(conditions.dtype),
            f"n{conditions.num_requests}",
            f"seed{conditions.seed}",
            slug(device),
        )
    )


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
        if path.name == MANIFEST:
            continue  # the folder's own index, not a trial in it
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


def trial_entry(
    trial: Trial, status: str, seconds: float | None = None, record: str | None = None
) -> dict:
    """One line of the index: a trial, how it ended, and where it landed."""
    return {
        "workload": trial.workload,
        "mode": trial.mode,
        "batch_size": trial.batch_size,
        "status": status,
        "seconds": seconds,
        "record": record,
    }


def read_manifest(run_dir: Path) -> dict:
    """The index a previous sweep over this folder left, or nothing."""
    try:
        index = json.loads((run_dir / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}  # no sweep here yet, or one that died mid-write
    return index if isinstance(index, dict) else {}


def banked_trials(index: dict) -> dict[Trial, dict]:
    """The entries an earlier index holds, keyed by the trial each describes."""
    banked = {}
    for entry in index.get("trials", []):
        try:
            banked[Trial(entry["workload"], entry["mode"], entry["batch_size"])] = entry
        except (TypeError, KeyError):
            continue  # not one of ours
    return banked


def manifest(
    sweep: str,
    args: argparse.Namespace,
    device: str,
    started: str,
    entries: list[dict],
    ceiling: dict[str, int],
) -> dict:
    """What the folder is, so it reads as one sweep rather than a pile of runs.

    Trials that were never attempted are the reason it is worth writing: they
    are the only outcome that leaves no record of its own, so without an index
    nothing on disk tells a size the ceiling ruled out from one that was never
    queued -- or from one a sweep died before reaching.
    """
    return {
        "sweep_id": sweep,
        "started": started,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": {
            "model": args.model,
            "dtype": args.dtype,
            "num_requests": args.num_requests,
            "seed": args.seed,
            "device": device,
        },
        "grid": {
            "workloads": args.workloads,
            "batch_sizes": sorted(args.batch_sizes),
            "sequential": args.sequential,
        },
        "ceiling": dict(sorted(ceiling.items())),
        "trials": entries,
    }


def write_manifest(run_dir: Path, index: dict) -> None:
    """Rewrite the folder's index.

    Called after every trial rather than at the end, because a sweep that dies
    part way through its grid is exactly the case an index is worth having,
    and one written on the way out is missing precisely then.
    """
    (run_dir / MANIFEST).write_text(json.dumps(index, indent=2) + "\n")


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


def run(args: argparse.Namespace, trial: Trial, run_dir: Path) -> str:
    """Run one trial in its own process: "ok", "oom", or "failed"."""
    command = [
        sys.executable,
        "-m",
        "benchmarks.throughput.bench",
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
        str(run_dir),
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
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="parent of the run folder"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="folder under --output-dir to gather this sweep's records in "
        "(default: named after the conditions, which is what lets a restart "
        "find them and resume)",
    )
    args = parser.parse_args(argv)

    unknown = set(args.workloads) - set(WORKLOADS)
    if unknown:
        parser.error(f"unknown workloads {sorted(unknown)}; known: {sorted(WORKLOADS)}")
    for size in args.batch_sizes:
        check_whole_waves(args.num_requests, size)

    fixed = Conditions(args.model, args.dtype, args.num_requests, args.seed)
    # Resolving the device here reads the driver's flag and, at most, asks
    # whether a GPU exists -- deliberately not `get_device_properties`, which
    # would open a CUDA context in *this* process and hold a few hundred MB on
    # the card for the whole sweep. That is memory the trials measuring their
    # own ceiling would no longer have, so the exact chip stays where it costs
    # nothing to read: the `environment` block of every record.
    device = loader.resolve_device(args.device)
    sweep = args.run_name or sweep_id(fixed, device)
    run_dir = args.output_dir / sweep
    run_dir.mkdir(parents=True, exist_ok=True)

    previous = read_manifest(run_dir)
    started = previous.get("started") or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    # A trial that ran under an earlier sweep keeps the entry it earned then:
    # the index describes the grid, not the invocation that happened to fill it
    # in, so resuming a finished sweep rewrites the same file.
    banked = banked_trials(previous)
    recorded = recorded_trials(run_dir)
    trials = plan(args)
    entries: dict[Trial, dict] = {}
    skipped, oom, unfit, failed = [], [], [], []
    ceiling: dict[str, int] = {}
    print(f"{len(trials)} trials -> {run_dir}")
    write_manifest(
        run_dir, manifest(sweep, args, device, started, list(banked.values()), ceiling)
    )
    for index, trial in enumerate(trials, start=1):
        progress = f"[{index}/{len(trials)}] {label(trial)}"
        status = recorded.get(trial_id(fixed, trial))
        if status is not None:
            print(f"{progress}: already recorded, skipping")
            skipped.append(trial)
            if status == "oom":
                note_ceiling(ceiling, trial)
            entries[trial] = banked.get(trial) or trial_entry(trial, status)
        elif over_ceiling(ceiling, trial):
            limit = ceiling[trial.workload]
            print(f"{progress}: not attempted, bs={limit} already ran out of memory")
            unfit.append(trial)
            entries[trial] = trial_entry(trial, "not attempted")
        else:
            print(progress)
            before = set(run_dir.glob("*.json"))
            start = time.perf_counter()
            outcome = run(args, trial, run_dir)
            seconds = time.perf_counter() - start
            # Whatever the trial wrote, including the record an out-of-memory
            # one leaves behind; a trial that died before writing has none.
            written = sorted(set(run_dir.glob("*.json")) - before)
            entries[trial] = trial_entry(
                trial, outcome, seconds, written[-1].name if written else None
            )
            if outcome == "oom":
                print(f"  OUT OF MEMORY: {label(trial)}; the ceiling for this workload")
                note_ceiling(ceiling, trial)
                oom.append(trial)
            elif outcome == "failed":
                print(f"  FAILED: {label(trial)}", file=sys.stderr)
                failed.append(trial)
        write_manifest(
            run_dir,
            manifest(sweep, args, device, started, list(entries.values()), ceiling),
        )

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
    print(f"records in {run_dir}")
    return exit_code(ran, len(failed))


if __name__ == "__main__":
    sys.exit(main())
