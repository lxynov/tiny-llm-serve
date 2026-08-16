import csv
import json

import pytest

from benchmarks.report import curves, main


def record(**overrides) -> dict:
    base = {
        "run_id": "20260812-000000-cuda-static-mixed-out-abc123",
        "model": "Qwen/Qwen3-8B",
        "device": "cuda",
        "dirty": False,
        "engine_mode": "static",
        "load": "offline-drain",
        "workload": "mixed-out",
        "num_requests": 128,
        "environment": {"gpu": "NVIDIA H100 80GB HBM3"},
        "config": {"dtype": "bfloat16", "batch_size": 8, "num_waves": 16},
        "metrics": {"output_tok_s": 1000.0, "s_per_decode_step": 0.01, "mbu": 0.4},
    }
    for key, value in overrides.items():
        if key in ("config", "metrics"):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def write(tmp_path, *records):
    for index, entry in enumerate(records):
        (tmp_path / f"{index}.json").write_text(json.dumps(entry))
    return tmp_path


def read_csv(tmp_path):
    with (tmp_path / "summary.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_records_flatten_into_one_row_each(tmp_path):
    write(
        tmp_path,
        record(config={"batch_size": 8}),
        record(config={"batch_size": 16}, metrics={"output_tok_s": 1500.0}),
    )

    assert main(["--records-dir", str(tmp_path), "--no-plots"]) == 0

    rows = read_csv(tmp_path)
    assert [r["batch_size"] for r in rows] == ["8", "16"]
    assert [r["output_tok_s"] for r in rows] == ["1000.0", "1500.0"]
    # Nested fields are lifted to columns so the CSV is usable as a table.
    assert rows[0]["gpu"] == "NVIDIA H100 80GB HBM3"
    assert rows[0]["dtype"] == "bfloat16"


def test_records_missing_a_metric_still_make_a_row(tmp_path):
    """A record written before a metric existed has a hole, not no row."""
    stale = record()
    del stale["metrics"]["mbu"]
    del stale["load"]
    write(tmp_path, stale)

    main(["--records-dir", str(tmp_path), "--no-plots"])

    (only,) = read_csv(tmp_path)
    assert only["mbu"] == ""
    assert only["load"] == ""
    assert only["output_tok_s"] == "1000.0"


def test_curves_group_by_workload_and_sort_by_batch_size():
    rows = [
        {"batch_size": 8, "workload": "mixed-out", "model": "m"},
        {"batch_size": 1, "workload": "mixed-out", "model": "m"},
        {"batch_size": 4, "workload": "uniform-512x128", "model": "m"},
        # The sequential baseline has no batch size and so no place on the axes.
        {"batch_size": None, "workload": "mixed-out", "model": "m"},
    ]

    grouped = curves(rows)

    assert list(grouped) == ["mixed-out", "uniform-512x128"]
    assert [r["batch_size"] for r in grouped["mixed-out"]] == [1, 8]


def test_curves_name_the_model_only_when_several_are_compared():
    rows = [
        {"batch_size": 1, "workload": "uniform-512x128", "model": "Qwen/Qwen3-8B"},
        {"batch_size": 1, "workload": "uniform-512x128", "model": "Qwen/Qwen3-0.6B"},
    ]

    assert set(curves(rows)) == {
        "Qwen3-8B uniform-512x128",
        "Qwen3-0.6B uniform-512x128",
    }


def test_plots_are_written_for_the_metrics_that_have_data(tmp_path):
    """An absent plot says no record carried that metric; an empty one would
    only look like a bug. These records have no memory peaks."""
    write(tmp_path, record(config={"batch_size": 8}), record(config={"batch_size": 16}))

    main(["--records-dir", str(tmp_path)])

    assert {p.name for p in tmp_path.glob("*.png")} == {
        "throughput.png",
        "decode_step_time.png",
        "mbu.png",
    }


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["--records-dir", str(tmp_path)])


def test_row_reports_a_dirty_tree(tmp_path, capsys):
    write(tmp_path, record(dirty=True))

    main(["--records-dir", str(tmp_path), "--no-plots"])

    assert "dirty tree" in capsys.readouterr().out
