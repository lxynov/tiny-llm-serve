import torch
import torch.nn.functional as F
from torch import nn


class KVCache:
    """Naive per-request KV cache.

    One contiguous [seq_len, num_kv_heads, head_dim] tensor pair per layer,
    grown by concatenation each step. To be replaced by a paged KV cache.
    """

    def __init__(self, num_layers: int) -> None:
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        return 0 if self._keys[0] is None else self._keys[0].shape[0]

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's k/v for a layer and return the full cached tensors."""
        if self._keys[layer_idx] is None:
            self._keys[layer_idx], self._values[layer_idx] = k, v
        else:
            self._keys[layer_idx] = torch.cat((self._keys[layer_idx], k))
            self._values[layer_idx] = torch.cat((self._values[layer_idx], v))
        return self._keys[layer_idx], self._values[layer_idx]


class Attention(nn.Module):
    """Causal attention via SDPA (FlashAttention on CUDA, math fallback on CPU)."""

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Attend q [q_len, num_heads, head_dim] to k/v [kv_len, num_kv_heads, head_dim].

        k/v include cached positions; num_kv_heads may divide num_heads (GQA).
        Returns [q_len, num_heads, head_dim].
        """
        q_len, kv_len = q.shape[0], k.shape[0]
        # SDPA's is_causal anchors the mask top-left, which is only correct when the
        # queries cover the whole sequence. The naive cache only ever needs full
        # prefill (q_len == kv_len) or single-token decode (attend to everything).
        if q_len > 1 and q_len != kv_len:
            raise ValueError(f"partial prefill unsupported: {q_len=} vs {kv_len=}")
        q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=q_len > 1, enable_gqa=True
        )
        return o.squeeze(0).transpose(0, 1)
