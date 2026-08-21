"""Aggregate a sweep folder into a report.

A sweep leaves one JSON record per trial next to an index over them: the shape
that is right to write and wrong to read. What the grid actually says is spread
across every file in the folder, and two of the numbers that say the most are
in none of them -- how much of the curve each batch size bought, and how much
of the batch was decoding sequences that had already finished.

This module joins the index to the records, derives those two, and prints a
table per workload. It also checks the thing the README's first rule asks of a
comparison and no single record can confirm: that every trial in the folder ran
the same commit on the same machine. A sweep resumed a week later against
rebuilt code looks exactly like one run in an afternoon -- until a curve bends
somewhere the engine did not change.

Nothing here re-derives a workload to get its lengths. The generator behind a
name can move, and a report that silently regenerates against today's version
would describe a grid that was never run; every column below comes from the
records themselves.

Usage (from the repo root):
    python -m benchmarks.report benchmarks/results/qwen3-8b-bfloat16-n512-seed0-cuda
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmarks import hardware
from benchmarks.sweep import MANIFEST, read_manifest

# The conditions rule 1 requires two trials to share before their numbers can
# be plotted against each other. `dtype` and `batch_size` live under `config`,
# so they are checked separately.
FIXED = ("model", "device", "num_requests", "seed", "commit")


@dataclass(frozen=True)
class Row:
    """One trial, with the columns no record carries computed in."""

    workload: str
    mode: str
    batch_size: int | None
    status: str
    wall_s: float | None = None
    prefill_share: float | None = None
    ms_per_step: float | None = None
    out_tok_s: float | None = None
    speedup: float | None = None
    occupancy: float | None = None
    kv_efficiency: float | None = None
    mbu: float | None = None
    bytes_s: float | None = None
    peak_gb: float | None = None

    @property
    def label(self) -> str:
        return "seq" if self.batch_size is None else str(self.batch_size)


def read_records(run_dir: Path) -> dict[str, dict]:
    """Every trial record in the folder, keyed by file name.

    Keyed by name because that is what the index refers to them by; a record
    the index does not mention still lands here, so a folder whose index died
    mid-write reports everything that ran.
    """
    records = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.name == MANIFEST:
            continue  # the folder's own index, not a trial in it
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue  # not one of ours, or written by a run that died
        if isinstance(record, dict) and "workload" in record:
            records[path.name] = record
    return records


def decode_occupancy(record: dict) -> float | None:
    """Fraction of decode slots that carried a sequence still generating.

    A static wave runs until its *longest* output finishes, so every sequence
    that finished earlier holds a slot in every step after it. The complement
    is the compute static batching spends on sequences that are already done --
    the waste `kv_efficiency` cannot see, since that one counts reserved KV
    rather than the steps run against it. The two come apart whenever prompts
    are more uniform than outputs: a reservation dominated by a fixed prompt
    looks efficient while most of the decoding is padding.

    Derived rather than measured: a wave's first token falls out of the prefill
    logits, so the step loop produced `output_tokens - num_requests` tokens
    against the `decode_steps x batch_size` slots it ran.
    """
    metrics = record.get("metrics") or {}
    steps, produced = metrics.get("decode_steps"), metrics.get("output_tokens")
    if not steps or produced is None:
        return None
    batch = record["config"].get("batch_size") or 1
    return (produced - record["num_requests"]) / (steps * batch)


def build_row(entry: dict, record: dict | None) -> Row:
    """One index entry, filled in from the record it points at."""
    base = dict(
        workload=entry["workload"],
        mode=entry["mode"],
        batch_size=entry["batch_size"],
        status=entry["status"],
    )
    metrics = (record or {}).get("metrics") or {}
    if not metrics:
        # An out-of-memory trial and one never attempted both land here: the
        # batch size is the result, and there are no metrics behind it.
        return Row(**base)
    wall, decode = metrics["wall_time_s"], metrics["decode_time_s"]
    steps = metrics["decode_steps"]
    peak = metrics.get("peak_gpu_memory_reserved_bytes")
    return Row(
        **base,
        wall_s=wall,
        prefill_share=metrics["prefill_time_s"] / wall if wall else None,
        ms_per_step=1000 * metrics["s_per_decode_step"] if steps else None,
        out_tok_s=metrics["output_tok_s"],
        occupancy=decode_occupancy(record) if record else None,
        kv_efficiency=metrics.get("kv_efficiency"),
        mbu=metrics.get("mbu"),
        bytes_s=metrics["decode_bytes_read"] / decode if decode else None,
        peak_gb=peak / 2**30 if peak else None,
    )


def with_speedup(rows: list[Row]) -> list[Row]:
    """Fill in each row's throughput against the cheapest trial that ran.

    Against the cheapest rather than against batch size 1, because a sweep
    restricted to large batches still has a baseline -- its own -- and one
    that skipped bs=1 would otherwise report no scaling at all.
    """
    ran = [r for r in rows if r.out_tok_s]
    if not ran:
        return rows
    base = min(ran, key=lambda r: r.batch_size or 0).out_tok_s
    return [
        r if not r.out_tok_s else Row(**{**vars(r), "speedup": r.out_tok_s / base})
        for r in rows
    ]


def build_rows(index: dict, records: dict[str, dict]) -> list[Row]:
    """The grid as rows, workload by workload, in the order the sweep ran it.

    Driven by the index rather than by the records, because the one outcome
    with no record of its own -- a batch size the ceiling ruled out before it
    was ever attempted -- is exactly the one a reader must not mistake for a
    size that was never in the grid.
    """
    entries = index.get("trials")
    if not entries:
        # No index, or one that died before its first write: fall back to what
        # the folder holds, which is every trial that actually ran.
        entries = [
            {
                "workload": r["workload"],
                "mode": r["engine_mode"],
                "batch_size": r["config"].get("batch_size"),
                "status": r.get("status", "ok"),
                "record": name,
            }
            for name, r in records.items()
        ]
    rows = [build_row(e, records.get(e.get("record") or "")) for e in entries]
    ordered: list[Row] = []
    for workload in dict.fromkeys(r.workload for r in rows):
        group = [r for r in rows if r.workload == workload]
        ordered += with_speedup(sorted(group, key=lambda r: r.batch_size or 0))
    return ordered


def inconsistencies(records: dict[str, dict]) -> list[str]:
    """Ways the folder breaks rule 1, one line each.

    A sweep folder is named after the conditions it holds fixed, so the name
    already promises most of this -- but the name is built from the *flags*,
    and neither the commit nor the machine is a flag. Resuming a sweep across
    an engine change is the easy way to get a curve whose bend belongs to the
    code rather than to the batch size, and nothing that writes here notices.
    """
    problems = []

    def disagree(name: str, values) -> None:
        # Compared as text: these are only ever reported, and a field that
        # changed type between records disagrees either way.
        seen = sorted({str(v) for v in values})
        if len(seen) > 1:
            problems.append(f"trials disagree on {name}: {', '.join(seen[:4])}")

    for field in FIXED:
        disagree(field, (r.get(field) for r in records.values()))
    disagree("dtype", (r.get("config", {}).get("dtype") for r in records.values()))
    disagree("gpu", ((r.get("environment") or {}).get("gpu") for r in records.values()))
    dirty = sorted(name for name, r in records.items() if r.get("dirty"))
    if dirty:
        problems.append(
            f"{len(dirty)} trial(s) measure uncommitted code, starting {dirty[0]}"
        )
    return problems


def fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


COLUMNS: list[tuple[str, str, str]] = [
    # header, attribute, format
    ("bs", "label", ""),
    ("status", "status", ""),
    ("wall_s", "wall_s", ".1f"),
    ("pre%", "prefill_share", ".1%"),
    ("ms/step", "ms_per_step", ".1f"),
    ("out_tok/s", "out_tok_s", ".1f"),
    ("speedup", "speedup", ".1f"),
    ("occ", "occupancy", ".3f"),
    ("kv_eff", "kv_efficiency", ".3f"),
    ("mbu", "mbu", ".3f"),
    ("peak_GB", "peak_gb", ".1f"),
]


def cells(row: Row) -> list[str]:
    out = []
    for _, attr, spec in COLUMNS:
        value = getattr(row, attr)
        out.append(value if isinstance(value, str) else fmt(value, spec))
    return out


def render(rows: list[Row], markdown: bool) -> list[str]:
    headers = [h for h, _, _ in COLUMNS]
    body = [cells(r) for r in rows]
    if markdown:
        return [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(r) + " |" for r in body),
        ]
    widths = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(headers)]
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    return [
        line,
        "-" * len(line),
        *("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in body),
    ]


def summarize(rows: list[Row], ceiling: int | None, peak_bytes_s: float | None) -> str:
    """What one workload's curve says, in the four numbers worth quoting."""
    ran = [r for r in rows if r.out_tok_s]
    if not ran:
        return "  no trial completed"
    best = max(ran, key=lambda r: r.out_tok_s or 0)
    lines = []
    smallest = min(ran, key=lambda r: r.batch_size or 0)
    linear = (best.batch_size or 1) / (smallest.batch_size or 1)
    scaling = f", {best.speedup / linear:.0%} of linear" if best.speedup else ""
    lines.append(
        f"  peak {best.out_tok_s:,.0f} out tok/s at bs={best.label} "
        f"({fmt(best.speedup, '.0f')}x the smallest batch{scaling})"
    )
    if len(ran) > 1:
        prev = ran[ran.index(best) - 1] if ran.index(best) else None
        if prev and prev.out_tok_s:
            gain = best.out_tok_s / prev.out_tok_s - 1
            lines.append(
                f"  last step bs={prev.label} -> {best.label} bought {gain:+.0%}"
            )
    band = [r.bytes_s for r in ran if r.bytes_s]
    if band:
        peak = f" of {peak_bytes_s / 1e12:.2f} TB/s peak" if peak_bytes_s else ""
        mbus = [r.mbu for r in ran if r.mbu]
        util = f", mbu {min(mbus):.2f}-{max(mbus):.2f}" if mbus else ""
        lines.append(
            f"  decode {min(band) / 1e9:,.0f}-{max(band) / 1e9:,.0f} GB/s{peak}{util}"
        )
    lines.append(
        f"  at bs={best.label}: occupancy {fmt(best.occupancy, '.2f')}, "
        f"kv efficiency {fmt(best.kv_efficiency, '.2f')}, "
        f"prefill {fmt(best.prefill_share, '.0%')} of wall"
    )
    if ceiling:
        lines.append(
            f"  ceiling: bs={ceiling} out of memory "
            f"(peak reserved {fmt(best.peak_gb, '.1f')} GB at bs={best.label})"
        )
    return "\n".join(lines)


def report(run_dir: Path, markdown: bool = False) -> str:
    index = read_manifest(run_dir)
    records = read_records(run_dir)
    if not records and not index.get("trials"):
        raise ValueError(f"{run_dir} holds no sweep records")
    rows = build_rows(index, records)
    conditions = index.get("conditions") or {}
    environment = next(iter(records.values()), {}).get("environment") or {}
    peak_bytes_s = hardware.peak_hbm_bytes_s(environment.get("gpu"))
    ceiling = index.get("ceiling") or {}

    out = [f"# {index.get('sweep_id', run_dir.name)}", ""]
    fixed = " ".join(f"{k}={v}" for k, v in conditions.items())
    out.append(fixed or "conditions: not indexed")
    commits = {str(r.get("commit")) for r in records.values()}
    commit = commits.pop()[:12] if len(commits) == 1 else "MIXED" if commits else "?"
    out.append(
        f"{environment.get('gpu', 'unknown device')}, torch "
        f"{environment.get('torch', '?')}, commit {commit}"
    )
    ok = sum(r.status == "ok" for r in rows)
    out.append(f"{ok}/{len(rows)} trials completed")
    for problem in inconsistencies(records):
        out.append(f"WARNING: {problem}")
    for workload in dict.fromkeys(r.workload for r in rows):
        group = [r for r in rows if r.workload == workload]
        out += ["", f"## {workload}", ""]
        out += render(group, markdown)
        out += ["", summarize(group, ceiling.get(workload), peak_bytes_s)]
    return "\n".join(out) + "\n"


def plot(run_dir: Path, out_dir: Path) -> list[Path]:
    """Three curves per sweep, one PNG each.

    Imported here rather than at the top of the module: plotting is the one
    thing this file does that needs a dependency the engine does not, and a
    report printed to a terminal should not pay for it.
    """
    import matplotlib

    matplotlib.use("Agg")  # a report is written to a file, never to a window
    import matplotlib.pyplot as plt

    rows = build_rows(read_manifest(run_dir), read_records(run_dir))
    panels = [
        ("throughput", "output tokens/s", "out_tok_s", True),
        ("decode-step", "ms per decode step", "ms_per_step", False),
        ("occupancy", "fraction of decode slots live", "occupancy", False),
    ]
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ylabel, attr, logy in panels:
        figure, axes = plt.subplots(figsize=(6, 4))
        for workload in dict.fromkeys(r.workload for r in rows):
            points = [
                (r.batch_size, getattr(r, attr))
                for r in rows
                if r.workload == workload and r.batch_size and getattr(r, attr)
            ]
            if points:
                axes.plot(*zip(*points), marker="o", label=workload)
        axes.set_xscale("log", base=2)
        if logy:
            axes.set_yscale("log")
        axes.set_xlabel("batch size")
        axes.set_ylabel(ylabel)
        axes.set_title(f"{run_dir.name}\n{ylabel} vs batch size", fontsize=9)
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8)
        figure.tight_layout()
        path = out_dir / f"{run_dir.name}-{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a sweep folder to report on")
    parser.add_argument(
        "--markdown", action="store_true", help="emit the tables as markdown"
    )
    parser.add_argument(
        "--plot",
        type=Path,
        metavar="DIR",
        help="also write throughput, decode-step and occupancy curves here",
    )
    args = parser.parse_args(argv)
    text = report(args.run_dir, markdown=args.markdown)
    if args.plot:
        for path in plot(args.run_dir, args.plot):
            text += f"wrote {path}\n"
    return text


if __name__ == "__main__":
    try:
        print(main(), end="")
    except (OSError, ValueError) as error:
        sys.exit(f"report: {error}")
