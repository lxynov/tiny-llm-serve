import json

from benchmarks.bench_throughput import main


def bench_args(tiny_checkpoint_path, tmp_path, **extra) -> list[str]:
    args = {
        "--model": str(tiny_checkpoint_path),
        "--workload": "uniform-512x128",
        "--num-requests": "2",
        "--repeats": "2",
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
    assert len(metrics["wall_times_s"]) == 2
    assert metrics["output_tok_s"] > 0
