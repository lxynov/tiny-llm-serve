import json

import pytest

from benchmarks import sweep
from benchmarks.sweep import Trial, exit_code, main, plan, recorded_trials


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


def test_sweep_runs_every_trial_in_its_own_process(tiny_checkpoint_path, tmp_path):
    assert main(sweep_args(tiny_checkpoint_path, tmp_path)) == 0

    records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert sorted(r["config"]["batch_size"] for r in records) == [1, 2]
    assert {r["engine_mode"] for r in records} == {"static"}


def test_recorded_trials_are_skipped_on_a_rerun(tiny_checkpoint_path, tmp_path):
    """Pods die mid-sweep; a restart should not re-buy what it already has."""
    main(sweep_args(tiny_checkpoint_path, tmp_path))
    first = {p.name for p in tmp_path.glob("*.json")}

    main(sweep_args(tiny_checkpoint_path, tmp_path))

    assert {p.name for p in tmp_path.glob("*.json")} == first


def test_a_failing_trial_does_not_end_the_sweep(tmp_path, capsys):
    """A trial that dies is not a reason to discard every trial still queued
    behind it."""
    # A checkpoint that cannot load stands in for a trial that dies on the GPU.
    args = sweep_args("/nonexistent-checkpoint", tmp_path, **{"--batch-sizes": "1,2"})

    assert main(args) == 1  # every trial failed, so the sweep itself is broken

    output = capsys.readouterr().out
    assert "[2/2]" in output  # the second trial was attempted regardless
    assert "0 run, 0 skipped, 0 out of memory, 0 not attempted, 2 failed" in output


def test_exit_code_distinguishes_a_ceiling_from_a_broken_sweep():
    assert exit_code(ran=6, failed=2) == 0  # most of the curve was still banked
    assert exit_code(ran=0, failed=8) == 1  # nothing ran at all
    assert exit_code(ran=8, failed=0) == 0


def oom_run(fails_at: int):
    """A `run` that reports the given batch size, and nothing else, as OOM."""
    attempted = []

    def run(args, trial: Trial) -> str:
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
    assert "OUT OF MEMORY: mixed-out static bs=4" in output
    assert "4 run, 0 skipped, 2 out of memory, 2 not attempted, 0 failed" in output


def test_a_recorded_ceiling_is_not_climbed_again_on_a_restart(monkeypatch, tmp_path):
    """The record left by the trial that ran out is what a resumed sweep reads
    to know where the curve ended; without it the restart pays for the sizes
    the first sweep already ruled out."""
    (tmp_path / "ceiling.json").write_text(
        json.dumps(
            {
                "model": "/unused-checkpoint",
                "num_requests": 8,
                "seed": 0,
                "workload": "mixed-out",
                "engine_mode": "static",
                "status": "oom",
                "config": {"dtype": "float32", "batch_size": 2},
            }
        )
    )
    run, attempted = oom_run(fails_at=None)
    monkeypatch.setattr(sweep, "run", run)
    args = sweep_args(
        "/unused-checkpoint",
        tmp_path,
        **{
            "--workloads": "mixed-out",
            "--batch-sizes": "1,2,4",
            "--num-requests": "8",
        },
    )

    assert main(args) == 0
    assert [t.batch_size for t in attempted] == [1]  # 2 is recorded, 4 is above it


def test_records_without_a_status_are_read_as_finished(tmp_path):
    """Every record written before the field existed came from a run that
    completed."""
    (tmp_path / "old.json").write_text(
        json.dumps(
            {
                "model": "m",
                "num_requests": 8,
                "seed": 0,
                "workload": "mixed-out",
                "engine_mode": "static",
                "config": {"dtype": "float32", "batch_size": 2},
            }
        )
    )

    assert list(recorded_trials(tmp_path).values()) == ["ok"]


def test_grid_that_cannot_divide_into_waves_is_rejected(tiny_checkpoint_path, tmp_path):
    """Caught before the first model load, not after every size that divided."""
    args = sweep_args(
        tiny_checkpoint_path,
        tmp_path,
        **{"--batch-sizes": "1,3", "--num-requests": "8"},
    )

    with pytest.raises(ValueError, match="not a multiple of --batch-size"):
        main(args)
    assert not list(tmp_path.glob("*.json"))


def test_plan_runs_small_batches_first_behind_the_baseline():
    """A sweep that dies at its ceiling should have banked the rest already."""

    class Args:
        workloads = ["mixed-out"]
        batch_sizes = [8, 1, 4]
        sequential = True

    assert plan(Args()) == [
        Trial("mixed-out", "sequential", None),
        Trial("mixed-out", "static", 1),
        Trial("mixed-out", "static", 4),
        Trial("mixed-out", "static", 8),
    ]


def test_unreadable_records_do_not_break_resumption(tmp_path):
    """A run killed mid-write leaves half a file; it should not be a trial."""
    (tmp_path / "truncated.json").write_text('{"model": "x",')

    assert recorded_trials(tmp_path) == {}
