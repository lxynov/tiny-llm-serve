import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Causal attention via SDPA (FlashAttention on CUDA, math fallback on CPU)."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend q [..., q_len, num_heads, head_dim] to k/v [..., kv_len,
        num_kv_heads, head_dim]; the leading batch dimension is optional.

        k/v include cached positions; num_kv_heads may divide num_heads (GQA).
        Without a mask, q_len == kv_len is causal prefill and q_len == 1 is
        single-token decode over the whole cache. A boolean `attn_mask`
        broadcastable to [..., 1, q_len, kv_len] (True = attend) covers
        everything else, e.g. batched decode over per-sequence valid lengths.
        Returns the same shape as q.
        """
        q_len, kv_len = q.shape[-3], k.shape[-3]
        # SDPA's is_causal anchors the mask top-left, which is only correct when
        # the queries cover the whole sequence (full prefill).
        if attn_mask is None and q_len > 1 and q_len != kv_len:
            raise ValueError(f"partial prefill needs a mask: {q_len=} vs {kv_len=}")
        batched = q.dim() == 4
        q, k, v = (x.transpose(-3, -2) for x in (q, k, v))
        if not batched:
            q, k, v = (x.unsqueeze(0) for x in (q, k, v))
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None and q_len > 1,
            enable_gqa=True,
        )
        if not batched:
            o = o.squeeze(0)
        return o.transpose(-3, -2)
