import torch


class NaiveKVCache:
    """Naive per-request KV cache.

    One contiguous [seq_len, num_kv_heads, head_dim] tensor pair per layer,
    grown by concatenation each step. To be replaced by a paged KV cache.
    """

    attn_mask: torch.Tensor | None = None

    def __init__(self, num_layers: int) -> None:
        self._keys: list[torch.Tensor | None] = [None] * num_layers
        self._values: list[torch.Tensor | None] = [None] * num_layers

    @property
    def seq_len(self) -> int:
        return 0 if self._keys[0] is None else self._keys[0].shape[0]

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append this step's k/v for a layer and return the full cached tensors."""
        cached_k, cached_v = self._keys[layer_idx], self._values[layer_idx]
        if cached_k is not None and cached_v is not None:
            # torch.cat allocates a new tensor and copies both inputs into it,
            # so each step re-copies the entire cache just to add one token:
            # O(seq_len) per step, O(n^2) over a decode. The preallocated
            # cache avoids this by reserving the full buffer up front and
            # writing each step in place.
            k = torch.cat((cached_k, k))
            v = torch.cat((cached_v, v))
        self._keys[layer_idx], self._values[layer_idx] = k, v
        return k, v
