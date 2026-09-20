"""The bytes decode has to move, whatever is timing it.

A duration means nothing on its own. Decode is memory-bandwidth-bound, so the
denominator that turns one into a rate is bytes -- all the weights once per
step, plus each sequence's KV window -- and dividing that rate by the device's
peak (`hardware.py`) turns it into the fraction of the machine actually in use.

Every benchmark charges the same bytes for the same work, which is what lets a
decode step timed in isolation be read against one timed inside a workload
pass.
"""

import torch

from tiny_llm_serve.config import ModelConfig


def weight_bytes(model: torch.nn.Module) -> int:
    """Bytes of parameters the device holds."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def kv_bytes_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    """Bytes one cached token occupies across every layer."""
    return (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * dtype.itemsize
    )


def decode_bytes_read(
    groups: list[tuple[int, int, int]], weights: int, kv_per_token: int
) -> int:
    """Bytes decode has to move, at minimum, to produce a pass's tokens.

    A group is (sequences, the KV length they start from, decode steps) --
    a static wave, a sequential request, or the window a microbenchmark timed.

    Every step re-reads all the weights to advance each sequence by one token,
    so the weight term is charged once per *step* rather than once per
    sequence. That is the batching win stated in bytes: it is why this total
    rises far more slowly than the batch size does, and why utilization climbs
    with it. On top of it each sequence reads its own KV window, which grows by
    one token every step.
    """
    total = 0
    for batch, kv_len, steps in groups:
        total += steps * weights
        window_tokens = steps * kv_len + steps * (steps + 1) // 2
        total += batch * window_tokens * kv_per_token
    return total
