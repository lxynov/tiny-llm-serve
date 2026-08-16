"""Turn a directory of benchmark records into a table and a few plots.

Records are one JSON file per run, which is the right shape for producing them
and the wrong shape for reading them. This flattens a directory of them into a
CSV and draws the plots a batch-size sweep exists to produce:

- **throughput vs batch size** -- the headline curve;
- **decode step time vs batch size** -- the same curve's explanation. It is
  roughly flat while host dispatch owns the step and rises once the step is
  genuinely bandwidth-bound, and the batch size where it turns is the point
  where batching stops being free;
- **MBU vs batch size**, against the 100% roofline -- how much of the machine
  is in use, and how much is left;
- **peak memory and KV efficiency vs batch size** -- what ends the sweep.

A batch size missing from a curve is one that did not fit: the sweep driver
leaves no record for a point that ran out of memory, so the end of a line is
the capacity ceiling.

Usage (from the repo root):
    python -m benchmarks.report --records-dir benchmarks/results/h100
"""

import argparse
import csv
import json
from pathlib import Path

from benchmarks.bench_throughput import RESULTS_DIR

COLUMNS = [
    "run_id",
    "date",
    "commit",
    "dirty",
    "model",
    "device",
    "gpu",
    "engine_mode",
    "load",
    "workload",
    "num_requests",
    "seed",
    "dtype",
    "batch_size",
    "num_waves",
    "wall_time_s",
    "wall_time_cv",
    "prefill_time_s",
    "decode_time_s",
    "decode_steps",
    "s_per_decode_step",
    "decode_bytes_read",
    "mbu",
    "weight_bytes",
    "kv_bytes_reserved",
    "peak_gpu_memory_allocated_bytes",
    "peak_gpu_memory_reserved_bytes",
    "peak_concurrent_seqs",
    "kv_efficiency",
    "prompt_tokens",
    "output_tokens",
    "output_tok_s",
    "total_tok_s",
]


def load_records(records_dir: Path) -> list[dict]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue  # not one of ours, or written by a run that died
    return records


def row(record: dict) -> dict:
    """One record flattened.

    Every field is looked up rather than indexed, so records written before a
    metric existed still make a row -- with a hole where that metric would be,
    which is the honest rendering of a run that never measured it.
    """
    metrics = record.get("metrics", {})
    config = record.get("config", {})
    flat = {
        "gpu": record.get("environment", {}).get("gpu"),
        "dtype": config.get("dtype"),
        "batch_size": config.get("batch_size"),
        "num_waves": config.get("num_waves"),
    }
    return {
        name: flat.get(name, record.get(name, metrics.get(name))) for name in COLUMNS
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def curves(rows: list[dict]) -> dict[str, list[dict]]:
    """Batched rows grouped into one line per workload, batch sizes ascending.

    Sequential rows have no batch size and so no place on these axes; they are
    the baseline the curves are read against, and they stay in the CSV. The
    model name joins the label only when a report spans more than one, which
    is what a model-scale contrast needs and what a single sweep does not.
    """
    batched = [r for r in rows if r["batch_size"] is not None]
    many_models = len({r["model"] for r in batched}) > 1
    grouped: dict[str, list[dict]] = {}
    for entry in batched:
        label = entry["workload"] or "?"
        if many_models:
            label = f"{Path(str(entry['model'])).name} {label}"
        grouped.setdefault(label, []).append(entry)
    return {
        label: sorted(points, key=lambda r: r["batch_size"])
        for label, points in sorted(grouped.items())
    }


def draw(
    axes, grouped: dict[str, list[dict]], column: str, ylabel: str, scale=1.0
) -> bool:
    """Draw one line per curve, and say whether anything was drawn."""
    drawn = False
    for label, points in grouped.items():
        pairs = [(p["batch_size"], p[column]) for p in points if p[column] is not None]
        if pairs:
            axes.plot(
                [x for x, _ in pairs],
                [y * scale for _, y in pairs],
                marker="o",
                label=label,
            )
            drawn = True
    axes.set_xscale("log", base=2)
    axes.set_xlabel("batch size")
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.3)
    if drawn:
        axes.legend(fontsize="small")
    return drawn


def write_plots(rows: list[dict], output_dir: Path) -> list[Path]:
    # Imported here so the CSV, which is the part that always works, does not
    # depend on a plotting library being installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = curves(rows)
    written = []
    for name, column, ylabel, scale in [
        ("throughput", "output_tok_s", "output tokens/s", 1.0),
        ("decode_step_time", "s_per_decode_step", "seconds per decode step", 1.0),
        ("mbu", "mbu", "memory bandwidth utilization", 1.0),
        (
            "peak_memory",
            "peak_gpu_memory_reserved_bytes",
            "peak reserved memory (GiB)",
            1 / 2**30,
        ),
        ("kv_efficiency", "kv_efficiency", "KV tokens used / reserved", 1.0),
    ]:
        figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
        if not draw(axes, grouped, column, ylabel, scale):
            # No record carried this metric -- a CPU run has no MBU, and a run
            # from before a metric existed has none of it. An absent plot says
            # that; an empty one only looks like a bug.
            plt.close(figure)
            continue
        if column in ("mbu", "kv_efficiency"):
            # The ceiling, drawn in: a utilization plot without it invites the
            # reader to judge the shape of the curve instead of its height.
            axes.axhline(1.0, linestyle="--", color="grey", linewidth=1)
            axes.set_ylim(0, 1.05)
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="default: --records-dir"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir or args.records_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.records_dir)
    if not records:
        parser.error(f"no records under {args.records_dir}")
    rows = [row(record) for record in records]

    csv_path = output_dir / "summary.csv"
    write_csv(rows, csv_path)
    print(f"wrote {csv_path} ({len(rows)} records)")
    dirty = sum(1 for r in rows if r["dirty"])
    if dirty:
        # Rule 4: a record from a dirty tree measures code no commit describes.
        print(f"warning: {dirty} record(s) came from a dirty tree")
    if not args.no_plots:
        for path in write_plots(rows, output_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
