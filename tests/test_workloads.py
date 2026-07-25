import pytest

from benchmarks.workloads import WORKLOADS, build_workload

VOCAB = 128


def test_same_seed_reproduces_workload():
    first = build_workload("sharegpt-like", 8, VOCAB, seed=0)
    second = build_workload("sharegpt-like", 8, VOCAB, seed=0)

    assert first == second


def test_different_seed_changes_workload():
    assert build_workload("sharegpt-like", 8, VOCAB, seed=0) != build_workload(
        "sharegpt-like", 8, VOCAB, seed=1
    )


def test_unknown_workload_raises():
    with pytest.raises(ValueError, match="unknown workload"):
        build_workload("nope", 1, VOCAB)


def test_uniform_lengths_are_fixed():
    requests = build_workload("uniform-512x128", 4, VOCAB)

    assert all(len(r.prompt_ids) == 512 for r in requests)
    assert all(r.output_len == 128 for r in requests)


def test_mixed_out_varies_only_output_lengths():
    requests = build_workload("mixed-out", 32, VOCAB)

    assert all(len(r.prompt_ids) == 512 for r in requests)
    assert all(16 <= r.output_len <= 512 for r in requests)
    assert len({r.output_len for r in requests}) > 1


def test_sharegpt_like_varies_both_lengths():
    requests = build_workload("sharegpt-like", 32, VOCAB)

    assert all(16 <= len(r.prompt_ids) <= 1024 for r in requests)
    assert all(16 <= r.output_len <= 1024 for r in requests)
    assert len({len(r.prompt_ids) for r in requests}) > 1


def test_prompt_ids_stay_within_vocab():
    for name in WORKLOADS:
        for request in build_workload(name, 4, VOCAB):
            assert all(0 <= t < VOCAB for t in request.prompt_ids)
