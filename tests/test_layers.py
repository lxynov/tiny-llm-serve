import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from tiny_llm_serve.layers.layernorm import RMSNorm

HIDDEN = 32


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
