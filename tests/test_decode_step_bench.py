import json
import re
from datetime import datetime

import pytest
import torch
from backends import backends

from benchmarks.decode_step.bench import (
    Cell,
    build_state,
    main,
    measure_cell,
    parse_cells,
    synthetic_prompts,
)
from benchmarks.records import code_version, cpu_name, git_state, hardware, slug
from benchmarks.roofline import decode_bytes_read, kv_bytes_per_token
from tiny_llm_serve.engine.llm_engine import LLM
from tiny_llm_serve.engine.sampling_params import SamplingParams

WARMUP, STEPS, REPEATS = 1, 2, 2
CPU = hardware("cpu", cpu_name())
CODE = code_version(*git_state())


def bench_args(tiny_checkpoint_path, tmp_path, **extra) -> list[str]:
    args = {
        "--model": str(tiny_checkpoint_path),
        "--cells": "1x8,2x12",
        "--warmup": str(WARMUP),
        "--steps": str(STEPS),
        "--repeats": str(REPEATS),
        "--dtype": "float32",
        "--device": "cpu",
        "--output-dir": str(tmp_path),
        **extra,
    }
    return [part for pair in args.items() for part in pair]


def test_parse_cells():
    assert parse_cells("1x576, 64x2048") == [Cell(1, 576), Cell(64, 2048)]
    for bad in ("", "64", "64x", "x576", "64*576", "0x576", "64x-1"):
        with pytest.raises(ValueError):
            parse_cells(bad)


def test_every_cell_is_measured_and_recorded(tiny_checkpoint_path, tmp_path):
    record = main(bench_args(tiny_checkpoint_path, tmp_path))

    (path,) = tmp_path.glob("*/*.json")
    assert json.loads(path.read_text()) == record
    assert path.stem == record["run_id"]
    # Filed with the runs it can be compared against: same day, commit, chip,
    # model and dtype -- and named for the time it started within that day.
    day = f"{datetime.fromisoformat(record['date']):%Y%m%d}"
    model = slug(tiny_checkpoint_path.name)
    assert path.parent.name == f"{day}-{CODE}-{CPU}-{model}-float32"
    assert re.fullmatch(r"\d{6}", record["run_id"])
    assert record["commit"] is not None
    assert record["config"]["cells"] == ["1x8", "2x12"]
    assert [c["cell"] for c in record["cells"]] == ["1x8", "2x12"]
    for cell in record["cells"]:
        metrics = cell["metrics"]
        assert cell["status"] == "ok"
        assert metrics["s_per_decode_step"] > 0
        assert metrics["bytes_s"] > 0
        # Off-GPU there is no peak bandwidth to divide by, and no allocator to
        # ask; a utilization against a guessed ceiling would be worse than none.
        assert metrics["mbu"] is None
        assert metrics["peak_gpu_memory_allocated_bytes"] is None
        assert len(cell["runs"]) == REPEATS


def test_recorded_bytes_match_the_window_that_was_timed(tiny_checkpoint_path, tmp_path):
    """The roofline is the throughput harness's own, charged over the steps
    this benchmark actually ran: all the weights once per step, plus each
    sequence's KV window, which grows by a token every step."""
    record = main(bench_args(tiny_checkpoint_path, tmp_path))

    llm = LLM(str(tiny_checkpoint_path), device="cpu")
    kv_per_token = kv_bytes_per_token(llm.model.config, torch.float32)
    timed = REPEATS * STEPS
    for cell in record["cells"]:
        # The timed steps center on the cell's kv_len: the middle one attends
        # exactly that many tokens.
        first = cell["metrics"]["kv_len_first_step"]
        assert first + (timed - 1) // 2 == cell["kv_len"]
        assert cell["metrics"]["kv_len_last_step"] == first + timed - 1
        cached = first - 1
        for run in cell["runs"]:
            assert run["decode_bytes_read"] == decode_bytes_read(
                [(cell["batch"], cached, STEPS)], record["weight_bytes"], kv_per_token
            )
            cached += STEPS


def test_a_cell_too_short_to_center_on_is_refused(tiny_checkpoint_path, tmp_path):
    """Its prefill would have to be empty or negative for the timed steps to
    center on its kv_len."""
    shortest = WARMUP + (REPEATS * STEPS + 1) // 2
    with pytest.raises(SystemExit):
        main(bench_args(tiny_checkpoint_path, tmp_path, **{"--cells": f"1x{shortest}"}))
    main(bench_args(tiny_checkpoint_path, tmp_path, **{"--cells": f"1x{shortest + 1}"}))


def test_a_cell_out_of_memory_does_not_end_the_run(
    tiny_checkpoint_path, tmp_path, monkeypatch, capsys
):
    """The batch size that did not fit is a result, and the cells queued behind
    it have not been asked yet."""

    def out_of_memory(llm, cell, *args):
        if cell.batch == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return measure_cell(llm, cell, *args)

    monkeypatch.setattr("benchmarks.decode_step.bench.measure_cell", out_of_memory)

    record = main(bench_args(tiny_checkpoint_path, tmp_path))

    first, second = record["cells"]
    assert first["status"] == "oom" and first["metrics"] == {}
    assert "out of memory" in first["error"].lower()
    assert second["status"] == "ok" and second["metrics"]["s_per_decode_step"] > 0
    assert "out of memory" in capsys.readouterr().err.lower()


@backends("all")
def test_the_batch_never_finishes_mid_run(tiny_checkpoint_path, device):
    """A finished batch turns a timed step into a no-op that still reports a
    time, so the budget and the slot both have to outlast the run."""
    llm = LLM(str(tiny_checkpoint_path), device=device)
    steps = 6
    state = build_state(llm, Cell(batch=2, kv_len=5), steps, seed=0)

    for _ in range(steps):
        assert not state.done
        llm.decode_step(state)

    assert state.steps == steps
    assert torch.equal(state.lens, torch.full_like(state.lens, 5 + steps))


def test_the_cache_is_random_where_a_correct_step_never_reads(tiny_checkpoint_path):
    """Zeros past a sequence's length are the worst thing to leave behind: a
    step that reads them drifts quietly. Random values make the same bug loud."""
    llm = LLM(str(tiny_checkpoint_path), device="cpu")
    written = 5
    cell = Cell(batch=2, kv_len=written)
    state = build_state(llm, cell, steps=2, seed=0)

    for cache in (state.manager.k_cache, state.manager.v_cache):
        for tensor in cache:
            assert (tensor[:, written:] != 0).all()

    # The prompt's own KV is what the prefill wrote, not scribble: filling it
    # too would make every measured step attend garbage. The same prompts
    # through a bare prefill are what that region has to still hold.
    bare = llm.prefill_batch(
        synthetic_prompts(cell, llm.model.config.vocab_size, 0),
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
        max_model_len=written,
    )
    assert torch.equal(
        state.manager.k_cache[0][:, :written],
        bare.manager.k_cache[0][:, :written],
    )
