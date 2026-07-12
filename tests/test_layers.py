import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from tiny_llm_serve.layers.layernorm import RMSNorm
from tiny_llm_serve.layers.rotary_embedding import RotaryEmbedding

HIDDEN = 32
INTERMEDIATE = 48
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


def tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
        max_position_embeddings=64,
    )


def test_rmsnorm_constant_input():
    norm = RMSNorm(4)
    norm.weight.data = torch.full((4,), 2.0)

    out = norm(torch.full((1, 4), 3.0))

    torch.testing.assert_close(out, torch.full((1, 4), 2.0), atol=1e-4, rtol=0)


def test_rmsnorm_matches_hf():
    torch.manual_seed(0)
    ours = RMSNorm(HIDDEN, eps=1e-6)
    ref = Qwen3RMSNorm(HIDDEN, eps=1e-6)
    weight = torch.randn(HIDDEN)
    ours.weight.data.copy_(weight)
    ref.weight.data.copy_(weight)
    x = torch.randn(5, HIDDEN)

    torch.testing.assert_close(ours(x), ref(x), atol=1e-4, rtol=1e-4)


def test_rotary_position_zero_is_identity():
    rope = RotaryEmbedding(HEAD_DIM, max_position_embeddings=64, base=1000000.0)
    q = torch.randn(1, NUM_HEADS, HEAD_DIM)
    k = torch.randn(1, NUM_KV_HEADS, HEAD_DIM)

    out_q, out_k = rope(torch.tensor([0]), q, k)

    torch.testing.assert_close(out_q, q, atol=1e-6, rtol=0)
    torch.testing.assert_close(out_k, k, atol=1e-6, rtol=0)


def test_rotary_preserves_norm():
    rope = RotaryEmbedding(HEAD_DIM, max_position_embeddings=64, base=1000000.0)
    q = torch.randn(3, NUM_HEADS, HEAD_DIM)
    k = torch.randn(3, NUM_KV_HEADS, HEAD_DIM)

    out_q, _ = rope(torch.tensor([7, 0, 42]), q, k)

    torch.testing.assert_close(out_q.norm(dim=-1), q.norm(dim=-1), atol=1e-4, rtol=1e-4)


def test_rotary_matches_hf():
    torch.manual_seed(0)
    positions = torch.tensor([3, 0, 7, 2, 5])
    seq_len = len(positions)
    q = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
    k = torch.randn(1, NUM_KV_HEADS, seq_len, HEAD_DIM)

    ref_rope = Qwen3RotaryEmbedding(tiny_qwen3_config())
    cos, sin = ref_rope(q, positions.unsqueeze(0))
    ref_q, ref_k = apply_rotary_pos_emb(q, k, cos, sin)

    rope = RotaryEmbedding(HEAD_DIM, max_position_embeddings=64, base=1000000.0)
    # Ours takes [num_tokens, heads, head_dim]; HF uses [batch, heads, seq, head_dim].
    out_q, out_k = rope(positions, q[0].transpose(0, 1), k[0].transpose(0, 1))

    torch.testing.assert_close(out_q.transpose(0, 1), ref_q[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_k.transpose(0, 1), ref_k[0], atol=1e-4, rtol=1e-4)
