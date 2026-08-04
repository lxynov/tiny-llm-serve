import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding, non-interleaved (Llama/Qwen convention).

    RoPE Implementation Formulas
    Let `d` be the embedding dimension (head_dim) and `m` be the token position.

    1. Frequencies:
       theta_i = 10000 ^ (-2i / d)  where i is in [0, 1, ..., d/2 - 1]

    2. Position Angles:
       angles = m * theta_i

    3. Split Vector (Non-interleaved):
       x_first = x[0 : d/2]
       x_second = x[d/2 : d]

    4. Rotate (Element-wise multiplication):
       y_first = (x_first * cos(angles)) - (x_second * sin(angles))
       y_second = (x_first * sin(angles)) + (x_second * cos(angles))

    5. Recombine:
       y = concatenate([y_first, y_second])
    """

    def __init__(
        self, head_dim: int, max_position_embeddings: int, base: float
    ) -> None:
        super().__init__()
        exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        theta = 1.0 / (base**exponents)
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        angles = torch.outer(positions, theta)
        # nn.Module.__getattr__ widens attribute access to Tensor | Module; annotate
        # the buffers so they stay tensors
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate q/k [..., num_heads, head_dim] by positions [...] (one
        position per token; any leading layout, e.g. flat or [batch, seq])."""
        # unsqueeze(-2) broadcasts [..., head_dim // 2] over the heads axis
        cos = self.cos_cache[positions].unsqueeze(-2).to(q.dtype)
        sin = self.sin_cache[positions].unsqueeze(-2).to(q.dtype)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_first, x_second = x.chunk(2, dim=-1)
    y_first = x_first * cos - x_second * sin
    y_second = x_first * sin + x_second * cos
    return torch.cat((y_first, y_second), dim=-1)
