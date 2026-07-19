import torch
from torch import nn

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.layers.attention import KVCache


class CausalLM(nn.Module):
    """Interface implemented by every model in MODEL_REGISTRY.

    Operates on a flattened [num_tokens] token layout (no batch dimension) and
    owns its own KV cache type, so the loader and engine can stay model-agnostic.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load an HF-format state dict into the model."""
        raise NotImplementedError

    def new_kv_cache(self) -> KVCache:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
