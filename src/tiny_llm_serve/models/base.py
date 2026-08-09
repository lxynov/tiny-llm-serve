import torch
from torch import nn

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.kv import KVCacheView


class CausalLM(nn.Module):
    """Interface implemented by every model in MODEL_REGISTRY.

    Operates on a flat [num_tokens] or padded [batch, seq] token layout and
    appends KV to whatever KVCacheView it is handed, so cache backends stay
    an engine concern and the loader and engine stay model-agnostic.

    `forward`'s optional `logits_indices` selects the positions the LM head
    scores -- token indices [k] in the flat layout, one per row [batch] in the
    padded one -- so prefill need not project a [..., seq, vocab_size] tensor
    to sample one row per sequence.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load an HF-format state dict into the model."""
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCacheView | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
