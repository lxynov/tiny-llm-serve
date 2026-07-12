import torch
from torch import nn


class MergedLinear(nn.Linear):
    """Projections sharing one input (Q/K/V, gate/up), fused into a single matmul.

    Checkpoints store the projections as separate tensors; `load_shard` copies
    each into its row slice of the fused weight.
    """

    def __init__(self, in_features: int, output_sizes: list[int]) -> None:
        super().__init__(in_features, sum(output_sizes), bias=False)
        self.output_sizes = output_sizes

    def load_shard(self, index: int, weight: torch.Tensor) -> None:
        start = sum(self.output_sizes[:index])
        self.weight.data[start : start + self.output_sizes[index]].copy_(weight)
