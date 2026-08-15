import json

import pytest

from benchmarks.sweep import Trial, exit_code, main, plan, recorded_trials


def sweep_args(tiny_checkpoint_path, tmp_path, **extra) -> list[str]:
    args = {
        "--model": str(tiny_checkpoint_path),
        "--workloads": "uniform-512x128",
        "--batch-sizes": "1,2",
        "--num-requests": "2",
        "--repeats": "1",
        "--warmup": "0",
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
    """Running out of memory at a large batch size is where capacity ran out,
    not a reason to discard every trial still queued behind it."""
    # A checkpoint that cannot load stands in for a trial that dies on the GPU.
    args = sweep_args("/nonexistent-checkpoint", tmp_path, **{"--batch-sizes": "1,2"})

    assert main(args) == 1  # every trial failed, so the sweep itself is broken

    output = capsys.readouterr().out
    assert "[2/2]" in output  # the second trial was attempted regardless
    assert "0 run, 0 skipped, 2 failed" in output


def test_exit_code_distinguishes_a_ceiling_from_a_broken_sweep():
    assert exit_code(ran=6, failed=2) == 0  # the largest batch sizes did not fit
    assert exit_code(ran=0, failed=8) == 1  # nothing ran at all
    assert exit_code(ran=8, failed=0) == 0


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

    assert recorded_trials(tmp_path) == set()
