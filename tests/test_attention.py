import pytest
import torch
from backends import backends

from tiny_llm_serve.layers.attention import Attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


@backends("all")
def test_attention_decode_matches_manual_gqa(device):
    torch.manual_seed(0)
    attn = Attention()
    q = torch.randn(1, NUM_HEADS, HEAD_DIM).to(device)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM).to(device)
    v = torch.randn(5, NUM_KV_HEADS, HEAD_DIM).to(device)

    out = attn(q, k, v)

    group_size = NUM_HEADS // NUM_KV_HEADS
    k_rep = k.repeat_interleave(group_size, dim=1)
    v_rep = v.repeat_interleave(group_size, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k_rep) / HEAD_DIM**0.5
    ref = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v_rep)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_attention_rejects_partial_prefill():
    attn = Attention()
    q = torch.randn(2, NUM_HEADS, HEAD_DIM)
    k = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(5, NUM_KV_HEADS, HEAD_DIM)

    with pytest.raises(ValueError, match="partial prefill"):
        attn(q, k, v)


@backends("all")
def test_batched_decode_mask_hides_invalid_keys(device):
    """A masked batched decode row equals an unbatched decode over only that
    row's valid prefix -- the property batched generation relies on."""
    torch.manual_seed(0)
    attn = Attention()
    q = torch.randn(2, 1, NUM_HEADS, HEAD_DIM, device=device)
    k = torch.randn(2, 5, NUM_KV_HEADS, HEAD_DIM, device=device)
    v = torch.randn(2, 5, NUM_KV_HEADS, HEAD_DIM, device=device)
    valid = torch.tensor([3, 5], device=device)
    mask = (torch.arange(5, device=device) < valid[:, None]).view(2, 1, 1, 5)

    out = attn(q, k, v, mask)

    for i, n in enumerate(valid.tolist()):
        ref = attn(q[i], k[i, :n], v[i, :n])
        torch.testing.assert_close(out[i], ref, atol=1e-5, rtol=1e-5)
