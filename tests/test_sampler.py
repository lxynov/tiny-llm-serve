import pytest
import torch

from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.layers.sampler import Sampler

# probabilities: [0.5, 0.3, 0.15, 0.05] (log gives exact softmax outputs)
NUCLEUS_LOGITS = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()


def seeded(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_default_params_is_greedy():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]])

    assert torch.equal(sampler(logits), torch.tensor([1, 0]))


def test_temperature_zero_is_greedy():
    sampler = Sampler()
    logits = torch.tensor([[0.1, 2.0, -1.0]])

    out = sampler(logits, SamplingParams(temperature=0.0))

    assert torch.equal(out, torch.tensor([1]))


def test_top_k_1_is_greedy():
    sampler = Sampler()
    logits = torch.randn(1, 16, generator=seeded())
    params = SamplingParams(temperature=1.0, top_k=1)

    outs = [sampler(logits, params, generator=seeded(i)) for i in range(20)]

    assert all(torch.equal(out, logits.argmax(dim=-1)) for out in outs)


def test_top_k_masks_tail():
    sampler = Sampler()
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    params = SamplingParams(temperature=1.0, top_k=2)
    generator = seeded()

    draws = {sampler(logits, params, generator=generator).item() for _ in range(50)}

    assert draws == {0, 1}


def test_top_p_keeps_nucleus():
    sampler = Sampler()
    # cumulative: 0.5, 0.8, 0.95, 1.0 -> top_p=0.7 keeps 0 and the crossing token 1
    params = SamplingParams(temperature=1.0, top_p=0.7)
    generator = seeded()

    draws = {
        sampler(NUCLEUS_LOGITS, params, generator=generator).item() for _ in range(100)
    }

    assert draws == {0, 1}


def test_top_p_and_temperature_one_leave_distribution_unchanged():
    sampler = Sampler()
    logits = torch.randn(1, 16, generator=seeded())
    params = SamplingParams(temperature=1.0, top_p=1.0)
    ours, ref = seeded(7), seeded(7)

    for _ in range(10):
        out = sampler(logits, params, generator=ours)
        expected = torch.multinomial(
            logits.softmax(dim=-1), num_samples=1, generator=ref
        ).squeeze(-1)
        assert torch.equal(out, expected)


def test_fixed_seed_is_reproducible():
    sampler = Sampler()
    logits = torch.randn(1, 16, generator=seeded())
    params = SamplingParams(temperature=1.0)

    a, b = seeded(42), seeded(42)
    draws_a = [sampler(logits, params, generator=a).item() for _ in range(20)]
    draws_b = [sampler(logits, params, generator=b).item() for _ in range(20)]

    assert draws_a == draws_b


def test_repetition_penalty_divides_positive_logits():
    sampler = Sampler()
    logits = torch.tensor([[2.0, 1.9, -3.0]])
    params = SamplingParams(temperature=0.0, repetition_penalty=1.5)
    seen = torch.tensor([[0]])

    out = sampler(logits, params, seen_token_ids=seen)

    # 2.0 / 1.5 = 1.33 < 1.9 -> the unseen token wins
    assert torch.equal(out, torch.tensor([1]))


def test_repetition_penalty_multiplies_negative_logits():
    sampler = Sampler()
    logits = torch.tensor([[-1.0, -1.05]])
    params = SamplingParams(temperature=0.0, repetition_penalty=1.5)
    seen = torch.tensor([[0]])

    out = sampler(logits, params, seen_token_ids=seen)

    # -1.0 * 1.5 = -1.5 < -1.05 -> the unseen token wins
    assert torch.equal(out, torch.tensor([1]))


def test_sampler_does_not_mutate_input_logits():
    sampler = Sampler()
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    original = logits.clone()
    params = SamplingParams(temperature=0.5, top_k=2, top_p=0.9, repetition_penalty=2.0)

    sampler(logits, params, seen_token_ids=torch.tensor([[0]]), generator=seeded())

    assert torch.equal(logits, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"top_k": 0},
        {"top_k": -2},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"repetition_penalty": 0.0},
        {"max_tokens": 0},
    ],
)
def test_sampling_params_validation(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)
