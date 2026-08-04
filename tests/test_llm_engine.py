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


@backends("cpu")
def test_pretokenized_prompt_matches_text_prompt(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    prompt_ids = llm.tokenizer.encode(PROMPT)

    from_ids = llm.generate_ids(prompt_ids, greedy(8))
    from_text = llm.generate_ids(PROMPT, greedy(8))

    assert from_ids == from_text


@backends("cpu")
def test_ignore_eos_generates_exactly_max_tokens(qwen3_path, device):
    """ignore_eos runs past the natural EOS but still respects max_tokens."""
    llm = load_llm(str(qwen3_path), device)
    prompt = resolve_prompt(llm.tokenizer, CHAT)  # a short answer, then EOS
    budget = 48

    baseline = llm.generate_ids(prompt, greedy(budget))
    forced = llm.generate_ids(prompt, greedy(budget, ignore_eos=True))

    assert len(baseline) < budget  # EOS actually fired within the budget
    assert len(forced) == budget


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


TINY_PROMPTS = [list(range(1, 9)), [3, 1, 4, 1, 5], [42], [7] * 6]

# Deliberately ragged (13/1/4/5 tokens). Batched decode pads every slot out to
# the longest row, so a short row reads past its own length -- into stale pad KV
# and never-written zeros -- only when the spread is wide. At near-equal lengths
# there is too little of that to flip a token, and dropping _DecodeStep.attn_mask
# entirely still passes. Keep these lengths far apart or this stops testing the
# mask.
RAGGED_PROMPTS = [
    PARITY_PROMPTS[3],  # 13 tokens
    "Hi",  # 1 token
    PARITY_PROMPTS[2],  # 4 tokens
    PARITY_PROMPTS[1],  # 5 tokens
]


@backends("all")
def test_static_batch_matches_sequential_on_tiny_model(tiny_checkpoint_path, device):
    """Day-9 definition of done: a static batch of 4 tiny-model sequences of
    different lengths and budgets produces exactly the tokens each would get
    alone.

    This pins slot isolation and per-row budgets -- the tiny model maps each
    prompt to its own constant token, so a row served another row's slot shows
    up immediately. It does not pin masking: random weights make the argmax
    indifferent to attention numerics, so a missing decode mask survives here
    at any length spread. test_static_batch_matches_sequential_on_qwen covers
    that.
    """
    llm = load_llm(str(tiny_checkpoint_path), device)
    budgets = [8, 12, 5, 12]
    params = [greedy(n) for n in budgets]

    batch = llm.generate_batch_ids(TINY_PROMPTS, params)
    sequential = [llm.generate_ids(p, greedy(n)) for p, n in zip(TINY_PROMPTS, budgets)]

    assert batch == sequential


@backends("all")
def test_static_batch_matches_sequential_on_qwen(qwen3_path, device):
    """Real-weight equivalence, where EOS can fire and rows finish at
    different steps while the rest of the batch continues.

    Weights sensitive enough to notice, over prompts ragged enough to expose
    what the decode mask hides: this is the test that fails if
    _DecodeStep.attn_mask stops masking.
    """
    llm = load_llm(str(qwen3_path), device)

    batch = llm.generate_batch(RAGGED_PROMPTS, greedy(10))
    sequential = [llm.generate(p, greedy(10)) for p in RAGGED_PROMPTS]

    assert batch == sequential


@backends("cpu")
def test_static_batch_seeded_sampling_is_reproducible(tiny_checkpoint_path, device):
    llm = load_llm(str(tiny_checkpoint_path), device)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8, seed=7)

    assert llm.generate_batch_ids(TINY_PROMPTS, params) == llm.generate_batch_ids(
        TINY_PROMPTS, params
    )


@backends("cpu")
def test_static_batch_raises_when_slots_overflow(tiny_checkpoint_path, device):
    llm = load_llm(str(tiny_checkpoint_path), device)

    with pytest.raises(RuntimeError, match="out of KV capacity"):
        llm.generate_batch_ids([[1, 2, 3, 4]], greedy(8), max_model_len=6)


def test_static_batch_rejects_unsupported_params(tiny_checkpoint_path):
    llm = load_llm(str(tiny_checkpoint_path), "cpu")

    with pytest.raises(ValueError, match="stop strings"):
        llm.generate_batch_ids([[1], [2]], greedy(4, stop=["x"]))
    with pytest.raises(ValueError, match="only in max_tokens"):
        llm.generate_batch_ids(
            [[1], [2]], [greedy(4), SamplingParams(temperature=0.8, max_tokens=4)]
        )
    with pytest.raises(ValueError, match="2 sampling params for 3"):
        llm.generate_batch_ids([[1], [2], [3]], [greedy(4), greedy(4)])
    with pytest.raises(ValueError, match="empty prompts"):
        llm.generate_batch_ids([[1], []], greedy(4))


@backends("all")
def test_seeded_sampling_is_reproducible(qwen3_path, device):
    llm = load_llm(str(qwen3_path), device)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=12, seed=42)

    first = llm.generate(PROMPT, params)
    second = llm.generate(PROMPT, params)

    assert first == second
