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
        """Attend q to k/v, whose leading kv_len - q_len positions are cached.

        Without a mask, q_len == kv_len is causal prefill and q_len == 1 is
        single-token decode over the whole cache. A boolean `attn_mask`
        (True = attend) covers everything else, e.g. batched decode over
        per-sequence valid lengths.

        Shapes:
            q:                     [*b, q_len, num_heads, head_dim]
            k, v:                  [*b, kv_len, num_kv_heads, head_dim]
            attn_mask:            ~[*b, 1, q_len, kv_len] bool | None
            q, after transpose:    [*b, num_heads, q_len, head_dim]
            k, v, after transpose: [*b, num_kv_heads, kv_len, head_dim]
            sdpa(q, k, v):         [batch, num_heads, q_len, head_dim]
            o, after squeeze:      [*b, num_heads, q_len, head_dim]
            ->                     same as q
          where kv_len >= q_len and num_heads % num_kv_heads == 0 -- GQA, with
          enable_gqa broadcasting each kv head across its query heads; an
          unbatched call is unsqueezed to batch 1 for SDPA and squeezed back
        """
        q_len, kv_len = q.shape[-3], k.shape[-3]
        # is_causal gives query i the keys 0..i, which are the right ones only if
        # the queries start at position 0; with a cache it would hide the past.
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
