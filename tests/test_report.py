import json

from benchmarks.throughput.report import (
    build_rows,
    decode_occupancy,
    inconsistencies,
    read_records,
    report,
)
from benchmarks.throughput.sweep import MANIFEST


def record(
    workload="mixed-out",
    batch_size=2,
    status="ok",
    output_tokens=6,
    num_requests=2,
    decode_steps=3,
    **extra,
) -> dict:
    """A trial record with only the fields the report reads."""
    body = {
        "model": "tiny",
        "device": "cpu",
        "commit": "abc123def456",
        "dirty": False,
        "seed": 0,
        "num_requests": num_requests,
        "workload": workload,
        "engine_mode": "static",
        "status": status,
        "config": {"dtype": "float32", "batch_size": batch_size},
        "environment": {"gpu": "NVIDIA H100 80GB HBM3", "torch": "2.13.0"},
        **extra,
    }
    if status == "ok":
        body["metrics"] = {
            "wall_time_s": 10.0,
            "prefill_time_s": 1.0,
            "decode_time_s": 9.0,
            "decode_steps": decode_steps,
            "s_per_decode_step": 9.0 / decode_steps,
            "output_tokens": output_tokens,
            "output_tok_s": output_tokens / 10.0,
            "decode_bytes_read": 9_000_000_000,
            "mbu": 0.2,
            "kv_efficiency": 0.5,
            "peak_gpu_memory_reserved_bytes": 2**30,
        }
    return body


def write_sweep(tmp_path, trials: list[tuple[dict, dict]], **index_extra):
    """A sweep folder: an index over trials, and a record for each that ran."""
    entries = []
    for entry, body in trials:
        name = f"{entry['workload']}-bs{entry['batch_size']}.json"
        if body is not None:
            (tmp_path / name).write_text(json.dumps(body))
        entries.append({**entry, "record": name if body is not None else None})
    (tmp_path / MANIFEST).write_text(
        json.dumps({"sweep_id": "test-sweep", "trials": entries, **index_extra})
    )
    return tmp_path


def entry(workload="mixed-out", batch_size=2, status="ok"):
    return {
        "workload": workload,
        "mode": "static",
        "batch_size": batch_size,
        "status": status,
    }


def test_decode_occupancy_counts_slots_held_by_finished_sequences():
    """Two sequences of 4 and 2 tokens: the wave runs 3 decode steps for the
    longer one, and the shorter holds a slot through the last two."""
    assert decode_occupancy(record(output_tokens=6, decode_steps=3)) == 4 / 6


def test_decode_occupancy_is_one_when_every_sequence_finishes_together():
    assert decode_occupancy(record(output_tokens=8, decode_steps=3)) == 1.0


def test_a_trial_the_ceiling_ruled_out_still_gets_a_row(tmp_path):
    """The one outcome with no record of its own must not read as a batch size
    that was never in the grid."""
    folder = write_sweep(
        tmp_path,
        [
            (entry(batch_size=1), record(batch_size=1)),
            (entry(batch_size=2, status="oom"), None),
            (entry(batch_size=4, status="not attempted"), None),
        ],
    )

    rows = build_rows(json.loads((folder / MANIFEST).read_text()), read_records(folder))

    assert [(r.batch_size, r.status) for r in rows] == [
        (1, "ok"),
        (2, "oom"),
        (4, "not attempted"),
    ]
    assert rows[1].out_tok_s is None


def test_speedup_is_measured_against_the_smallest_batch_that_ran(tmp_path):
    folder = write_sweep(
        tmp_path,
        [
            (entry(batch_size=4), record(batch_size=4, output_tokens=10)),
            (entry(batch_size=8), record(batch_size=8, output_tokens=30)),
        ],
    )

    rows = build_rows(json.loads((folder / MANIFEST).read_text()), read_records(folder))

    assert [r.speedup for r in rows] == [1.0, 3.0]


def test_a_folder_without_an_index_still_reports_what_ran(tmp_path):
    (tmp_path / "trial.json").write_text(json.dumps(record()))

    assert "mixed-out" in report(tmp_path)


def test_trials_from_different_commits_are_flagged():
    """Rule 1 asks that a sweep hold the engine fixed, and the folder's own
    name -- built from flags -- cannot promise it."""
    records = {"a.json": record(), "b.json": record(commit="999999999999")}

    assert any("commit" in problem for problem in inconsistencies(records))


def test_a_trial_measuring_uncommitted_code_is_flagged():
    problems = inconsistencies({"a.json": record(dirty=True)})

    assert problems and "uncommitted" in problems[0]


def test_a_consistent_sweep_is_flagged_for_nothing():
    assert inconsistencies({"a.json": record(), "b.json": record(batch_size=4)}) == []


def test_markdown_renders_a_table(tmp_path):
    folder = write_sweep(tmp_path, [(entry(), record())])

    assert "| bs | status |" in report(folder, markdown=True)
