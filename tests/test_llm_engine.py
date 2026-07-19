from functools import lru_cache
from pathlib import Path

import pytest
import torch
from backends import backends
from transformers import AutoModelForCausalLM

from tiny_llm_serve.engine.llm_engine import LLM
from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.models import loader

PROMPT = "Where is Winterfell?"
CHAT = "<chat>"  # sentinel: render CHAT_MESSAGE through the model's chat template
CHAT_MESSAGE = "What is the capital of France?"
PARITY_PROMPTS = [
    PROMPT,
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    CHAT,
]
PARITY_MAX_TOKENS = 16


def greedy(max_tokens: int, **kwargs) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **kwargs)


@lru_cache(maxsize=None)
def load_llm(model_path: str, device: str, dtype: torch.dtype = torch.float32) -> LLM:
    """Cache one LLM per (checkpoint, backend, dtype) so parametrized tests reuse it."""
    return LLM(model_path, device=device, dtype=dtype)


@lru_cache(maxsize=None)
def load_reference(model_path: str, device: str, dtype: torch.dtype):
    """Cache the HuggingFace model our greedy decoding is checked against."""
    return (
        AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).eval().to(device)
    )


@lru_cache(maxsize=None)
def load_tokenizer(model_path: str):
    """The tokenizer alone, for decoding a failure report without a model."""
    return loader.load_tokenizer(Path(model_path))


def resolve_prompt(tokenizer, prompt: str) -> str:
    """Expand the CHAT sentinel via the chat template; pass raw prompts through."""
    if prompt != CHAT:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": CHAT_MESSAGE}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def greedy_parity(
    model_path: str, device: str, dtype: torch.dtype, prompt: str
) -> tuple[list[int], list[int]]:
    """Greedy-decode `prompt` with both engines; return (ours, HF's) token ids."""
    llm = load_llm(model_path, device, dtype)
    text = resolve_prompt(llm.tokenizer, prompt)
    ours = llm.generate_ids(text, greedy(PARITY_MAX_TOKENS))

    eos = llm.tokenizer.eos_token_id
    input_ids = torch.tensor([llm.tokenizer.encode(text)], device=device)
    with torch.no_grad():
        ref = (
            load_reference(model_path, device, dtype)
            .generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=PARITY_MAX_TOKENS,
                do_sample=False,
                eos_token_id=eos,  # align the stop rule: ours stops only on this id
                pad_token_id=eos,
            )[0, input_ids.shape[1] :]
            .tolist()
        )
    if ref and ref[-1] == eos:
        ref = ref[:-1]  # ours excludes the EOS token
    return ours, ref


def common_prefix_len(a: list[int], b: list[int]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


def divergence_report(model_path: str, ours: list[int], ref: list[int]) -> str:
    """Where the two decodings split, and what each produced from there."""
    tokenizer = load_tokenizer(model_path)
    lcp = common_prefix_len(ours, ref)
    return (
        f"diverged after {lcp}/{max(len(ours), len(ref))} tokens\n"
        f"  ours: {tokenizer.decode(ours[lcp : lcp + 8])!r}\n"
        f"  ref:  {tokenizer.decode(ref[lcp : lcp + 8])!r}"
    )


@backends("all")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("prompt", PARITY_PROMPTS)
def test_parity_with_hf_greedy(qwen3_path, device, prompt, dtype):
    """Greedy decoding matches HF token for token, in both dtypes.

    bfloat16 is the tighter of the two: a near-tie can flip a token and send
    the two decodings apart for good. So read a failure that shows up only in
    bf16 as a question -- which token flipped, and was it close? -- before
    assuming a bug. The report on the assertion answers that.
    """
    ours, ref = greedy_parity(str(qwen3_path), device, dtype, prompt)

    assert ours == ref, divergence_report(str(qwen3_path), ours, ref)


@backends("all")
def test_generate_stops_before_stop_substring(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    baseline = llm.generate(PROMPT, greedy(20))
    assert len(baseline) > 4

    # A slice of the (deterministic) greedy output is guaranteed to appear,
    # so the stop condition is certain to trigger.
    stop_str = baseline[len(baseline) // 2 : len(baseline) // 2 + 3]

    output = llm.generate(PROMPT, greedy(20, stop=[stop_str]))

    assert stop_str not in output
    assert len(output) < len(baseline)


@backends("all")
def test_generate_ids_stops_decoding_at_stop_substring(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    baseline_ids = llm.generate_ids(PROMPT, greedy(20))
    baseline = llm.tokenizer.decode(baseline_ids)
    assert len(baseline) > 4

    stop_str = baseline[len(baseline) // 2 : len(baseline) // 2 + 3]

    output_ids = llm.generate_ids(PROMPT, greedy(20, stop=[stop_str]))

    # The decode loop must stop at the stop string, not run out `max_tokens`.
    assert 0 < len(output_ids) < len(baseline_ids)


@backends("all")
def test_seeded_sampling_is_reproducible(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12, seed=42)

    first = llm.generate(PROMPT, params)
    second = llm.generate(PROMPT, params)

    assert first == second
