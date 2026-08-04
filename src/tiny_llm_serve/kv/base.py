"""The two contracts every KV-cache backend implements.

KVCacheView is the per-step surface a decoder layer sees; KVManager is the
per-request surface the engine sees. Splitting them keeps models ignorant of
admission policy: a model only ever appends to whatever view it is handed.
"""

from typing import Protocol

import torch


class KVCacheView(Protocol):
    """What a decoder layer needs from any KV-cache backend during one forward
    pass: somewhere to append this step's k/v, and the attention mask describing
    which cached positions are valid. `attn_mask` follows Attention's
    convention: None when causal masking suffices, otherwise a boolean mask
    broadcastable to [..., 1, seq_len, kv_len], True where attention may look."""

    attn_mask: torch.Tensor | None

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store one layer's k/v for this step, [..., seq_len, num_kv_heads,
        head_dim], and return the k/v attention should read over:
        [..., kv_len, num_kv_heads, head_dim] with kv_len >= seq_len, cached
        positions included and padded to the longest sequence in the step.
        Pass the result to Attention together with `attn_mask`, which says
        which of those positions are real.

        The leading batch dimension is present exactly when the backend batches
        sequences. A view describes a single step, so call this once per layer
        per forward pass.
        """
        ...


class KVManager:
    """Admission control + KV storage, implemented per backend.

    Backends differ in what `admit` reserves: a preallocating backend claims a
    whole max_model_len slot up front (Orca/FasterTransformer-style); a paged
    backend claims blocks incrementally. The engine sees the same surface
    either way.
    """

    def can_admit(self, num_prompt_tokens: int) -> bool:
        raise NotImplementedError

    def admit(self, num_prompt_tokens: int) -> int:
        """Reserve capacity for a new sequence; return the slot that holds it."""
        raise NotImplementedError

    def free(self, slot: int) -> None:
        raise NotImplementedError

    def begin_prefill(self, slots: list[int], prompt_lens: list[int]) -> KVCacheView:
        raise NotImplementedError

    def begin_decode(self, slots: list[int]) -> KVCacheView:
        raise NotImplementedError
