"""Batch-size sweep driver.

Walks a workload x batch-size grid, running each trial as a separate
`bench_throughput` process and leaving the records behind. Four properties
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
- **One folder per sweep.** Records land together in a directory under
  `--output-dir`, next to a `sweep.json` index of the grid. The folder is
  named for the day the sweep started and the commit it measures, then for
  what it held fixed -- the chip included, not just the kind of device. A sweep
  never writes into a folder that already exists, so two runs of one grid
  cannot pass for one curve.

Usage (from the repo root):
    python -m benchmarks.throughput.sweep --model Qwen/Qwen3-8B --dtype bfloat16 \
        --num-requests 128 --batch-sizes 1,2,4,8,16,32,64,128
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.records import (
    DTYPES,
    OOM_EXIT,
    code_version,
    cpu_name,
    git_state,
    hardware,
    slug,
)
from benchmarks.throughput.bench import (
    MODES,
    OUTPUT_DIR,
    check_whole_waves,
    trial_name,
)
from benchmarks.throughput.workloads import WORKLOADS
from tiny_llm_serve.models import loader

# The folder's index, and the one file in it that is not a trial.
MANIFEST = "sweep.json"


@dataclass(frozen=True)
class Trial:
    workload: str
    batch_size: int


def probe_chip(device: str) -> str | None:
    """The name of the chip `device` runs on, as a record's `environment` block
    would report it.

    A CUDA device is asked in a throwaway process: reading the name here would
    open a CUDA context in *this* one and hold a few hundred MB on the card for
    the whole sweep -- memory the trials measuring their own ceiling would no
    longer have. The child inherits the same `CUDA_VISIBLE_DEVICES` the trials
    do, so `cuda:1` names the card they will actually land on. MPS and CPU runs
    are bound by the SoC or CPU itself, which costs nothing to read.
    """
    if not loader.is_cuda(device):
        return cpu_name()
    probe = "import sys, torch; print(torch.cuda.get_device_name(sys.argv[1]))"
    return subprocess.check_output(
        [sys.executable, "-c", probe, device], text=True
    ).strip()


def sweep_id(
    started: datetime,
    code: str,
    hardware: str,
    model: str,
    dtype: str,
    num_requests: int,
) -> str:
    """The folder this sweep's records land in.

    It leads with the day the sweep started and the code it measures, as
    `code_version` spells it, so a listing of `--output-dir` sorts into a
    history of the engine. After them come the chip and what the sweep held
    fixed, so the name alone says whether two sweeps can be plotted against
    each other -- the chip included, which the record format otherwise asks
    you to keep apart by hand. The day is taken in UTC, like every timestamp in
    the records, and from the start, so a sweep that runs past midnight keeps
    one name.
    """
    return "-".join(
        (
            f"{started:%Y%m%d}",
            code,
            hardware,
            slug(Path(model).name),
            slug(dtype),
            f"n{num_requests}",
        )
    )


def trial_entry(
    trial: Trial, status: str, seconds: float | None = None, record: str | None = None
) -> dict:
    """One line of the index: a trial, how it ended, and where it landed."""
    return {
        "workload": trial.workload,
        "batch_size": trial.batch_size,
        "status": status,
        "seconds": seconds,
        "record": record,
    }


def read_manifest(run_dir: Path) -> dict:
    """The index the sweep left in this folder, or nothing."""
    try:
        index = json.loads((run_dir / MANIFEST).read_text())
    except (OSError, ValueError):
        return {}  # no index here, or one cut off mid-write
    return index if isinstance(index, dict) else {}


def manifest(
    sweep: str,
    args: argparse.Namespace,
    device: str,
    chip: str | None,
    commit: str | None,
    dirty: bool | None,
    started: datetime,
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
        "started": started.isoformat(timespec="seconds"),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": {
            "model": args.model,
            "dtype": args.dtype,
            "num_requests": args.num_requests,
            "seed": args.seed,
            "device": device,
            "chip": chip,
            "commit": commit,
            "dirty": dirty,
        },
        "grid": {
            "workloads": args.workloads,
            "batch_sizes": sorted(args.batch_sizes),
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
    return [
        Trial(workload, size)
        for workload in args.workloads
        for size in sorted(args.batch_sizes)
    ]


def label(trial: Trial) -> str:
    return f"{trial.workload} bs={trial.batch_size}"


def exit_code(ran: int, failed: int) -> int:
    """Nonzero only when the sweep itself is broken. Trials that ran out of
    memory measured their ceiling, so they count as neither."""
    return 1 if failed and ran == 0 else 0


def note_ceiling(ceiling: dict[str, int], trial: Trial) -> None:
    """Remember the smallest batch size that did not fit on this workload."""
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
    return limit is not None and trial.batch_size >= limit


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
        "--batch-size",
        str(trial.batch_size),
        "--num-requests",
        str(args.num_requests),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(run_dir),
    ]
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
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="default: auto-select")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="parent of the run folder"
    )
    args = parser.parse_args(argv)

    unknown = set(args.workloads) - set(WORKLOADS)
    if unknown:
        parser.error(f"unknown workloads {sorted(unknown)}; known: {sorted(WORKLOADS)}")
    for size in args.batch_sizes:
        check_whole_waves(args.num_requests, size)

    # Resolving the device here reads the driver's flag and, at most, asks
    # whether a GPU exists; the chip comes from `probe_chip`, which keeps any
    # CUDA context out of this process.
    device = loader.resolve_device(args.device)
    chip = probe_chip(device)
    commit, dirty = git_state()
    started = datetime.now(timezone.utc)
    sweep = sweep_id(
        started,
        code_version(commit, dirty),
        hardware(device, chip),
        args.model,
        args.dtype,
        args.num_requests,
    )
    run_dir = args.output_dir / sweep
    if run_dir.exists():
        # Trial files are named for the trial alone, so a second run of the
        # grid would overwrite the first one record at a time, leaving a mix
        # that nothing on disk could tell apart.
        parser.error(f"{run_dir} already exists; move it aside to rerun this sweep")
    run_dir.mkdir(parents=True)

    trials = plan(args)
    entries: list[dict] = []
    oom, unfit, failed = [], [], []
    ceiling: dict[str, int] = {}

    def save_manifest() -> None:
        write_manifest(
            run_dir,
            manifest(
                sweep, args, device, chip, commit, dirty, started, entries, ceiling
            ),
        )

    print(f"{len(trials)} trials -> {run_dir}")
    save_manifest()
    for index, trial in enumerate(trials, start=1):
        progress = f"[{index}/{len(trials)}] {label(trial)}"
        if over_ceiling(ceiling, trial):
            limit = ceiling[trial.workload]
            print(f"{progress}: not attempted, bs={limit} already ran out of memory")
            unfit.append(trial)
            entries.append(trial_entry(trial, "not attempted"))
        else:
            print(progress)
            start = time.perf_counter()
            outcome = run(args, trial, run_dir)
            seconds = time.perf_counter() - start
            # Whatever the trial wrote, including the record an out-of-memory
            # one leaves behind; a trial that died before writing has none.
            name = f"{trial_name(MODES[-1], trial.workload, trial.batch_size)}.json"
            written = name if (run_dir / name).exists() else None
            entries.append(trial_entry(trial, outcome, seconds, written))
            if outcome == "oom":
                print(f"  OUT OF MEMORY: {label(trial)}; the ceiling for this workload")
                note_ceiling(ceiling, trial)
                oom.append(trial)
            elif outcome == "failed":
                print(f"  FAILED: {label(trial)}", file=sys.stderr)
                failed.append(trial)
        save_manifest()

    ran = len(trials) - len(oom) - len(unfit) - len(failed)
    print(
        f"\n{ran} run, {len(oom)} out of memory, "
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
