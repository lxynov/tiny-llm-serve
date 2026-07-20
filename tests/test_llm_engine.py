from functools import lru_cache

import torch
from backends import backends
from transformers import AutoModelForCausalLM

from tiny_llm_serve.engine.llm_engine import LLM

PROMPT = "Where is Winterfell?"


@lru_cache(maxsize=None)
def load_llm(model_path: str, device: str) -> LLM:
    """Cache one LLM per (checkpoint, backend) so parametrized tests reuse it."""
    return LLM(model_path, device=device)


@backends("all")
def test_generate_ids_matches_hf_greedy(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    input_ids = llm.tokenizer.encode(PROMPT)
    ref_model = (
        AutoModelForCausalLM.from_pretrained(qwen3_path, dtype=torch.float32)
        .eval()
        .to(device)
    )
    with torch.no_grad():
        ref_ids = ref_model.generate(
            torch.tensor([input_ids], device=device),
            max_new_tokens=10,
            do_sample=False,
        )[0, len(input_ids) :]

    output_ids = llm.generate_ids(PROMPT, max_tokens=10)

    assert output_ids == ref_ids.tolist()


@backends("all")
def test_generate_stops_before_stop_substring(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    baseline = llm.generate(PROMPT, max_tokens=20)
    assert len(baseline) > 4

    # A slice of the (deterministic) greedy output is guaranteed to appear,
    # so the stop condition is certain to trigger.
    stop_str = baseline[len(baseline) // 2 : len(baseline) // 2 + 3]

    output = llm.generate(PROMPT, max_tokens=20, stop=[stop_str])

    assert stop_str not in output
    assert len(output) < len(baseline)


@backends("all")
def test_generate_ids_stops_decoding_at_stop_substring(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    baseline_ids = llm.generate_ids(PROMPT, max_tokens=20)
    baseline = llm.tokenizer.decode(baseline_ids)
    assert len(baseline) > 4

    stop_str = baseline[len(baseline) // 2 : len(baseline) // 2 + 3]

    output_ids = llm.generate_ids(PROMPT, max_tokens=20, stop=[stop_str])

    # The decode loop must stop at the stop string, not run out `max_tokens`.
    assert 0 < len(output_ids) < len(baseline_ids)
