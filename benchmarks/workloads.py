"""Deterministic benchmark workloads.

A workload is a seeded list of requests, each a random-token prompt plus a
forced output length. Running it with `ignore_eos` makes output token counts
exact, so the same workload produces identical work across engine modes and
deltas between runs are attributable to the engine, not the input.
"""

import math
import random
from dataclasses import dataclass


@dataclass
class Request:
    prompt_ids: list[int]
    output_len: int


def _uniform_512x128(rng: random.Random) -> tuple[int, int]:
    """Fixed 512-token prompts and 128-token outputs: static batching's best
    case, since every sequence finishes on the same step."""
    return 512, 128


def _mixed_out(rng: random.Random) -> tuple[int, int]:
    """Fixed 512-token prompts with long-tailed output lengths in [16, 512]:
    under static batching every sequence waits for the longest output."""
    return 512, _lognormal(rng, median=64, low=16, high=512)


def _sharegpt_like(rng: random.Random) -> tuple[int, int]:
    """Long-tailed prompt and output lengths in [16, 1024], approximating the
    shape of ShareGPT traces (most requests short, a heavy tail of long ones)."""
    return (
        _lognormal(rng, median=128, low=16, high=1024),
        _lognormal(rng, median=128, low=16, high=1024),
    )


def _lognormal(rng: random.Random, median: int, low: int, high: int) -> int:
    return min(max(int(rng.lognormvariate(math.log(median), 1.0)), low), high)


WORKLOADS = {
    "uniform-512x128": _uniform_512x128,
    "mixed-out": _mixed_out,
    "sharegpt-like": _sharegpt_like,
}


def build_workload(
    name: str, num_requests: int, vocab_size: int, seed: int = 0
) -> list[Request]:
    """Generate `num_requests` requests for the named workload, reproducibly.

    Prompts are uniform-random token ids in [0, vocab_size): throughput does
    not depend on prompt content, and random ids keep the harness independent
    of any tokenizer or dataset.
    """
    if name not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; known: {sorted(WORKLOADS)}")
    rng = random.Random(seed)
    requests = []
    for _ in range(num_requests):
        prompt_len, output_len = WORKLOADS[name](rng)
        prompt_ids = [rng.randrange(vocab_size) for _ in range(prompt_len)]
        requests.append(Request(prompt_ids, output_len))
    return requests
