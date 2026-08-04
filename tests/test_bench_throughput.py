import json

from benchmarks.bench_throughput import main
from benchmarks.workloads import build_workload


def bench_args(tiny_checkpoint_path, tmp_path, **extra) -> list[str]:
    args = {
        "--model": str(tiny_checkpoint_path),
        "--workload": "uniform-512x128",
        "--num-requests": "2",
        "--device": "cpu",
        "--output-dir": str(tmp_path),
        **extra,
    }
    return [part for pair in args.items() for part in pair]


def test_harness_writes_a_complete_record(tiny_checkpoint_path, tmp_path):
    record = main(bench_args(tiny_checkpoint_path, tmp_path))

    (path,) = tmp_path.glob("*.json")
    assert json.loads(path.read_text()) == record
    assert path.stem == record["run_id"]
    assert "-cpu-sequential-uniform-512x128-" in record["run_id"]
    assert record["engine_mode"] == "sequential"
    assert record["workload"] == "uniform-512x128"
    assert record["commit"] is not None
    environment = record["environment"]
    assert environment["torch"] and environment["cpu"]
    assert environment["torch_threads"] >= 1
    metrics = record["metrics"]
    # ignore_eos forces exact lengths: 2 requests x 512 prompt, 128 output
    assert metrics["prompt_tokens"] == 2 * 512
    assert metrics["output_tokens"] == 2 * 128
    assert metrics["wall_time_s"] > 0
    assert metrics["output_tok_s"] > 0
    # The naive sequential cache reserves exactly what it stores.
    assert metrics["kv_efficiency"] == 1.0
    assert metrics["peak_concurrent_seqs"] == 1


def test_static_mode_drains_the_workload_in_waves(tiny_checkpoint_path, tmp_path):
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{
                "--workload": "mixed-out",
                "--num-requests": "3",
                "--mode": "static",
                "--batch-size": "2",
            },
        )
    )

    assert record["engine_mode"] == "static"
    assert record["config"]["batch_size"] == 2
    metrics = record["metrics"]
    # All 3 requests ran (two waves of 2 + 1), with forced output lengths.
    workload = build_workload("mixed-out", 3, 128, seed=0)
    assert metrics["output_tokens"] == sum(r.output_len for r in workload)
    assert metrics["peak_concurrent_seqs"] == 2
    # Slots reserve the wave's worst case, so mixed lengths waste reservation.
    assert 0 < metrics["kv_efficiency"] < 1
