import json
from datetime import datetime, timezone

import pytest

from benchmarks.records import code_version, cpu_name, git_state, hardware
from benchmarks.throughput import bench, sweep
from benchmarks.throughput.sweep import (
    MANIFEST,
    Trial,
    exit_code,
    main,
    plan,
    sweep_id,
)

# What every test sweep runs on, as its folder name spells it.
CPU = hardware("cpu", cpu_name())
# The code every test sweep measures: this checkout, as it stands.
CODE = code_version(*git_state())


def sweep_args(tiny_checkpoint_path, tmp_path, **extra) -> list[str]:
    args = {
        "--model": str(tiny_checkpoint_path),
        "--workloads": "uniform-512x128",
        "--batch-sizes": "1,2",
        "--num-requests": "2",
        "--device": "cpu",
        "--output-dir": str(tmp_path),
        **extra,
    }
    return [part for pair in args.items() for part in pair]


def run_dir(tmp_path):
    """The one folder a sweep into `tmp_path` gathered its records in."""
    (folder,) = tmp_path.iterdir()
    return folder


def records(run_dir):
    return [
        json.loads(p.read_text()) for p in run_dir.glob("*.json") if p.name != MANIFEST
    ]


def test_sweep_runs_every_trial_in_its_own_process(tiny_checkpoint_path, tmp_path):
    assert main(sweep_args(tiny_checkpoint_path, tmp_path)) == 0

    trials = records(run_dir(tmp_path))
    assert sorted(r["config"]["batch_size"] for r in trials) == [1, 2]
    assert {r["engine_mode"] for r in trials} == {"static"}


def test_a_failing_trial_does_not_end_the_sweep(tmp_path, capsys):
    """A trial that dies is not a reason to discard every trial still queued
    behind it."""
    # A checkpoint that cannot load stands in for a trial that dies on the GPU.
    args = sweep_args("/nonexistent-checkpoint", tmp_path, **{"--batch-sizes": "1,2"})

    assert main(args) == 1  # every trial failed, so the sweep itself is broken

    output = capsys.readouterr().out
    assert "[2/2]" in output  # the second trial was attempted regardless
    assert "0 run, 0 out of memory, 0 not attempted, 2 failed" in output


def test_exit_code_distinguishes_a_ceiling_from_a_broken_sweep():
    assert exit_code(ran=6, failed=2) == 0  # most of the curve was still banked
    assert exit_code(ran=0, failed=8) == 1  # nothing ran at all
    assert exit_code(ran=8, failed=0) == 0


def oom_run(fails_at: int | None):
    """A `run` that reports the given batch size, and nothing else, as OOM."""
    attempted = []

    def run(args, trial: Trial, run_dir) -> str:
        attempted.append(trial)
        return "oom" if trial.batch_size == fails_at else "ok"

    return run, attempted


def test_a_batch_size_that_does_not_fit_ends_its_own_ladder(
    monkeypatch, tmp_path, capsys
):
    """Memory demand only rises with the batch size, so every size above one
    that ran out is hopeless -- and confirming that costs a model load apiece.
    The other workload's curve is unaffected: ceilings are per workload."""
    run, attempted = oom_run(fails_at=4)
    monkeypatch.setattr(sweep, "run", run)
    args = sweep_args(
        "/unused-checkpoint",
        tmp_path,
        **{
            "--workloads": "mixed-out,uniform-512x128",
            "--batch-sizes": "1,2,4,8",
            "--num-requests": "8",
        },
    )

    assert main(args) == 0  # a ceiling is a result, not a broken sweep

    for workload in ("mixed-out", "uniform-512x128"):
        sizes = [t.batch_size for t in attempted if t.workload == workload]
        assert sizes == [1, 2, 4]  # 8 was never attempted
    output = capsys.readouterr().out
    assert "OUT OF MEMORY: mixed-out bs=4" in output
    assert "4 run, 2 out of memory, 2 not attempted, 0 failed" in output


def test_a_rerun_does_not_write_into_an_earlier_sweeps_folder(
    monkeypatch, tmp_path, capsys
):
    """Trial files are named for the trial alone, so a second run of the same
    grid would overwrite the first one record at a time -- and leave a folder
    that reads as one curve while holding two."""
    run, attempted = oom_run(fails_at=None)
    monkeypatch.setattr(sweep, "run", run)
    args = sweep_args("/unused-checkpoint", tmp_path)
    main(args)
    index = (run_dir(tmp_path) / MANIFEST).read_text()

    with pytest.raises(SystemExit):
        main(args)

    assert "already exists" in capsys.readouterr().err
    assert len(attempted) == 2  # the first sweep's two trials, and no more
    assert (run_dir(tmp_path) / MANIFEST).read_text() == index


def test_grid_that_cannot_divide_into_waves_is_rejected(tiny_checkpoint_path, tmp_path):
    """Caught before the first model load, not after every size that divided."""
    args = sweep_args(
        tiny_checkpoint_path,
        tmp_path,
        **{"--batch-sizes": "1,3", "--num-requests": "8"},
    )

    with pytest.raises(ValueError, match="not a multiple of --batch-size"):
        main(args)
    assert not list(tmp_path.iterdir())  # not even the folder


def test_plan_runs_small_batches_first():
    """A sweep that dies at its ceiling should have banked the rest already."""

    class Args:
        workloads = ["mixed-out"]
        batch_sizes = [8, 1, 4]

    assert plan(Args()) == [
        Trial("mixed-out", 1),
        Trial("mixed-out", 4),
        Trial("mixed-out", 8),
    ]


def test_a_second_engine_mode_has_to_reach_the_sweep_key():
    """`Trial` keys on (workload, batch size) alone, which is only safe while
    one engine mode exists: the sweep names each record after the default mode
    and never passes `--mode`, so a second one would be run under the first's
    name. If this fails, a mode was added -- put it in `Trial`, in
    `trial_entry`, and on the `--mode` the sweep passes to each trial."""
    assert bench.MODES == ["static"]


def test_sweep_id_leads_with_the_day_and_the_code():
    """So a listing of the results folder sorts into a history of the engine,
    and the rest of the name says whether two sweeps can be plotted together."""
    h100 = hardware("cuda", "NVIDIA H100 80GB HBM3")
    started = datetime(2026, 8, 19, 18, 53, 20, tzinfo=timezone.utc)

    assert (
        sweep_id(started, "faa6722", h100, "Qwen/Qwen3-8B", "bfloat16", 512)
        == "20260819-faa6722-cuda-h100-80gb-hbm3-qwen3-8b-bfloat16-n512"
    )
    # A dot is legal in a path and load-bearing in a model name.
    assert sweep_id(
        started, "faa6722", h100, "Qwen/Qwen3-0.6B", "bfloat16", 512
    ).endswith("-qwen3-0.6b-bfloat16-n512")


def test_code_version_keeps_uncommitted_runs_apart():
    """A dirty run is not the commit it names, so it must not land among that
    commit's clean records."""
    commit = "faa672214e4fa9959d68bc2c95645d3c3abc47c6"

    assert code_version(commit, False) == "faa6722"
    assert code_version(commit, True) == "faa6722-dirty"
    assert code_version(None, None) == "unknown-commit"


def test_hardware_names_the_chip_not_just_the_kind_of_device():
    # Which card is not what the card is.
    assert hardware("cuda:1", "NVIDIA H100 PCIe") == "cuda-h100-pcie"
    # One Mac, two backends: the chip alone would file them together.
    assert hardware("mps", "Apple M3 Max") == "mps-apple-m3-max"
    assert hardware("cpu", "Apple M3 Max") == "cpu-apple-m3-max"
    assert hardware("cpu", None) == "cpu-unknown"


def test_a_sweep_gathers_its_records_in_one_folder(tiny_checkpoint_path, tmp_path):
    assert main(sweep_args(tiny_checkpoint_path, tmp_path)) == 0

    folder = run_dir(tmp_path)
    index = json.loads((folder / MANIFEST).read_text())
    # Named for the day the sweep started, which the index records.
    started = datetime.fromisoformat(index["started"])
    model = str(tiny_checkpoint_path)
    assert folder.name == sweep_id(started, CODE, CPU, model, "bfloat16", 2)
    trials = ["static-uniform-512x128-bs001.json", "static-uniform-512x128-bs002.json"]
    assert sorted(p.name for p in folder.iterdir()) == [*trials, MANIFEST]
    assert [t["record"] for t in index["trials"]] == trials


def test_a_sweep_that_cannot_be_compared_gets_its_own_folder(monkeypatch, tmp_path):
    """Two request counts are two workloads' worth of work, not two points on
    one curve, so their records must not land in the same pile."""
    run, _ = oom_run(fails_at=None)
    monkeypatch.setattr(sweep, "run", run)

    for num_requests in ("2", "4"):
        main(
            sweep_args(
                "/unused-checkpoint", tmp_path, **{"--num-requests": num_requests}
            )
        )

    assert {p.name.split("-")[-1] for p in tmp_path.iterdir()} == {"n2", "n4"}


def test_the_index_holds_the_trials_that_left_no_record(monkeypatch, tmp_path):
    """A trial the ceiling ruled out is the one outcome with nothing on disk of
    its own; without the index a reader cannot tell it from one never queued."""
    run, _ = oom_run(fails_at=2)
    monkeypatch.setattr(sweep, "run", run)
    args = sweep_args(
        "/unused-checkpoint",
        tmp_path,
        **{"--batch-sizes": "1,2,4", "--num-requests": "4"},
    )

    main(args)

    folder = run_dir(tmp_path)
    index = json.loads((folder / MANIFEST).read_text())
    assert index["sweep_id"] == folder.name
    assert index["conditions"]["device"] == "cpu"
    assert index["conditions"]["chip"] == cpu_name()
    assert index["conditions"]["commit"] == git_state()[0]
    assert index["grid"]["batch_sizes"] == [1, 2, 4]
    assert index["ceiling"] == {"uniform-512x128": 2}
    assert [(t["batch_size"], t["status"]) for t in index["trials"]] == [
        (1, "ok"),
        (2, "oom"),
        (4, "not attempted"),
    ]
