import torch
import torch.nn.functional as F
from torch import nn


class SiluAndMul(nn.Module):
    """SwiGLU activation over an input holding [gate, up] on the last dim."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shapes: x: [..., 2 * intermediate_size] -> [..., intermediate_size]"""
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
