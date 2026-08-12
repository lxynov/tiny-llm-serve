import json

import pytest
import torch

from benchmarks.bench_throughput import (
    decode_bytes_read,
    decode_groups,
    main,
    weight_bytes,
)
from benchmarks.workloads import build_workload


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
    # The naive sequential cache reserves exactly what it stores.
    assert metrics["kv_efficiency"] == 1.0
    assert metrics["peak_concurrent_seqs"] == 1
    # Both memory peaks are recorded, and both are null off-GPU.
    assert metrics["peak_gpu_memory_allocated_bytes"] is None
    assert metrics["peak_gpu_memory_reserved_bytes"] is None


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
                "--repeats": "1",
                "--warmup": "0",
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


def test_phases_are_timed_separately_and_account_for_the_pass(
    tiny_checkpoint_path, tmp_path
):
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{
                "--num-requests": "2",
                "--mode": "static",
                "--batch-size": "2",
                "--repeats": "1",
                "--warmup": "0",
            },
        )
    )

    metrics = record["metrics"]
    # One wave of 128-token outputs: the prefill logits yield the first token,
    # so the remaining 127 come from decode forward passes.
    assert metrics["decode_steps"] == 127
    assert metrics["prefill_time_s"] > 0
    assert metrics["decode_time_s"] > 0
    assert metrics["s_per_decode_step"] == pytest.approx(metrics["decode_time_s"] / 127)
    # The two phases partition the pass; only bookkeeping falls between them.
    phases = metrics["prefill_time_s"] + metrics["decode_time_s"]
    assert phases == pytest.approx(metrics["wall_time_s"], rel=0.05)


def test_sequential_mode_times_every_request(tiny_checkpoint_path, tmp_path):
    """Timing accumulates across calls, so one pass sums all of its requests."""
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{"--num-requests": "2", "--repeats": "1", "--warmup": "0"},
        )
    )

    metrics = record["metrics"]
    # Two requests, each running its 128 decode passes back to back.
    assert metrics["decode_steps"] == 2 * 128
    phases = metrics["prefill_time_s"] + metrics["decode_time_s"]
    assert phases == pytest.approx(metrics["wall_time_s"], rel=0.05)


def test_byte_accounting_matches_the_closed_form(tiny_checkpoint_path, tmp_path):
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{
                "--num-requests": "2",
                "--mode": "static",
                "--batch-size": "2",
                "--repeats": "1",
                "--warmup": "0",
            },
        )
    )

    metrics = record["metrics"]
    # The tiny config: 2 layers x 2 KV heads x 8 head_dim, keys and values,
    # float32 -- 256 bytes per cached token.
    per_token = 2 * 2 * 2 * 8 * 4
    # One wave of 2 slots, each reserving 512 prompt + 128 output tokens.
    assert metrics["kv_bytes_reserved"] == 2 * (512 + 128) * per_token
    assert metrics["weight_bytes"] > 0


@pytest.mark.parametrize("mode, batch_size", [("static", "2"), ("sequential", "8")])
def test_decode_step_model_matches_what_the_engine_ran(
    tiny_checkpoint_path, tmp_path, mode, batch_size
):
    """The roofline counts bytes from a model of the decode loop; if that model
    disagreed with the loop, MBU would be wrong in a way nothing else shows."""
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{
                "--workload": "mixed-out",
                "--num-requests": "4",
                "--mode": mode,
                "--batch-size": batch_size,
                "--repeats": "1",
                "--warmup": "0",
            },
        )
    )

    requests = build_workload("mixed-out", 4, 128, seed=0)
    groups = decode_groups(requests, mode, int(batch_size))
    assert sum(steps for _, _, steps in groups) == record["metrics"]["decode_steps"]


def test_decode_bytes_charge_the_weights_once_per_step():
    """The batching win in bytes: sixteen sequences do not read sixteen copies
    of the weights, so the byte total grows far slower than the batch."""
    one = decode_bytes_read([(1, 0, 4)], weights=1000, kv_per_token=0)
    sixteen = decode_bytes_read([(16, 0, 4)], weights=1000, kv_per_token=0)

    assert one == sixteen == 4 * 1000


def test_decode_bytes_count_each_sequences_growing_window():
    """Four steps from a window of 10 read 11 + 12 + 13 + 14 tokens."""
    assert decode_bytes_read([(2, 10, 4)], weights=0, kv_per_token=1) == 2 * 50


def test_mbu_is_null_without_a_known_ceiling(tiny_checkpoint_path, tmp_path):
    record = main(bench_args(tiny_checkpoint_path, tmp_path, **{"--repeats": "1"}))

    assert record["metrics"]["decode_bytes_read"] > 0
    assert record["metrics"]["mbu"] is None


def test_record_describes_the_workload_it_measured(tiny_checkpoint_path, tmp_path):
    record = main(
        bench_args(
            tiny_checkpoint_path,
            tmp_path,
            **{"--workload": "mixed-out", "--num-requests": "8", "--repeats": "1"},
        )
    )

    assert record["load"] == "offline-drain"
    stats = record["workload_stats"]
    # mixed-out holds prompts at 512 and varies only the output length.
    assert stats["prompt_len"] == {"mean": 512, "median": 512, "min": 512, "max": 512}
    requests = build_workload("mixed-out", 8, 128, seed=0)
    lengths = [r.output_len for r in requests]
    assert stats["output_len"]["min"] == min(lengths)
    assert stats["output_len"]["max"] == max(lengths)
    # The tail is the point of this workload: the longest output sets how long
    # every sequence in its wave is held open.
    assert stats["output_len"]["max"] > stats["output_len"]["median"]


def test_weight_bytes_counts_tied_storage_once():
    """Tied embeddings are one storage under two names -- and one HBM cost."""
    shared = torch.nn.Linear(4, 4, bias=False)
    tied = torch.nn.Linear(4, 4, bias=False)
    tied.weight = shared.weight

    assert weight_bytes(torch.nn.Sequential(shared, tied)) == 4 * 4 * 4
