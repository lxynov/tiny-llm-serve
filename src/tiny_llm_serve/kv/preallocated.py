"""Slot-reserving KV manager for batched generation.

A manager owns the physical cache tensors (data plane) and the slot
bookkeeping (control plane). Admission is the only memory decision: an
admitted sequence reserves everything it could ever need up front, so it can
never outgrow its reservation mid-flight. Models never touch the manager
directly -- each engine step starts with begin_prefill/begin_decode, which
returns a view speaking the KVCacheView protocol.
"""

import torch

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.kv.base import KVManager


class PreallocatedKVManager(KVManager):
    """A fixed pool of sequence slots, each reserving `max_model_len` tokens of
    contiguous KV per layer.

    The reservation makes admitted sequences safe by construction but wastes
    every slot token past a sequence's true length -- the waste the
    kv_efficiency metric measures and paged attention will reclaim.

    Shapes:
        k_cache[layer], v_cache[layer]:
            [num_slots, max_model_len, num_kv_heads, head_dim]
        cached_seq_lens: [num_slots] int64
    """

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int,
        max_model_len: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.max_model_len = max_model_len
        self.device = device
        shape = (num_slots, max_model_len, config.num_key_value_heads, config.head_dim)
        self.k_cache = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        # Cached tokens per slot.
        self.cached_seq_lens = torch.zeros(num_slots, dtype=torch.long, device=device)
        self._cached_seq_lens_host = [0] * num_slots
        self._free_slots = list(range(num_slots))
        self._last_slot_ids = torch.zeros(0, dtype=torch.long, device=device)
        self._last_slot_ids_host: list[int] = []

    def can_admit(self, num_prompt_tokens: int) -> bool:
        return bool(self._free_slots) and num_prompt_tokens <= self.max_model_len

    def admit(self, num_prompt_tokens: int) -> int:
        if not self.can_admit(num_prompt_tokens):
            raise ValueError(
                f"cannot admit a {num_prompt_tokens}-token prompt: "
                f"{len(self._free_slots)} free slots of {self.max_model_len} tokens"
            )
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        if slot in self._free_slots:
            raise ValueError(f"slot {slot} is already free")
        self.cached_seq_lens[slot] = 0
        self._cached_seq_lens_host[slot] = 0
        self._free_slots.append(slot)

    def begin_prefill(self, slots: list[int], prompt_lens: list[int]) -> "_PrefillStep":
        if any(self._cached_seq_lens_host[slot] for slot in slots):
            raise ValueError(f"prefill into occupied slots {slots}")
        if max(prompt_lens) > self.max_model_len:
            raise ValueError(f"prompt of {max(prompt_lens)} tokens exceeds a slot")
        for slot, prompt_len in zip(slots, prompt_lens):
            self._cached_seq_lens_host[slot] = prompt_len
        slot_ids = self._update_slot_ids_if_changed(slots)
        self.cached_seq_lens[slot_ids] = torch.tensor(prompt_lens, device=self.device)
        return _PrefillStep(self, slot_ids)

    def begin_decode(self, slots: list[int]) -> "_DecodeStep":
        cached = [self._cached_seq_lens_host[slot] for slot in slots]
        if not all(cached):
            raise ValueError(f"decode from unprefilled slots {slots}")
        kv_len = max(cached) + 1
        if kv_len > self.max_model_len:
            raise RuntimeError(
                f"out of KV capacity: a slot reached {self.max_model_len} tokens"
            )
        for slot in slots:
            self._cached_seq_lens_host[slot] += 1
        slot_ids = self._update_slot_ids_if_changed(slots)
        write_pos = self.cached_seq_lens[slot_ids]
        self.cached_seq_lens[slot_ids] = write_pos + 1
        return _DecodeStep(self, slot_ids, write_pos, kv_len)

    def _update_slot_ids_if_changed(self, slots: list[int]) -> torch.Tensor:
        """The device copy of `slots`, rebuilt only when the batch changes.

        Rebuilding is a blocking host-to-device copy, and on CUDA a blocking copy
        waits for all queued GPU work to finish -- a sync. A static batch passes
        the same slots every step, so it pays that once, at prefill. The change
        check compares against a host copy of the last slots, since reading the
        device tensor back would sync too.

        Shapes:
            -> [batch] int64
        """
        if slots != self._last_slot_ids_host:
            self._last_slot_ids_host = list(slots)
            self._last_slot_ids = torch.tensor(slots, device=self.device)
        return self._last_slot_ids


class _PrefillStep:
    """Writes a right-padded prompt block into fresh slots.

    attn_mask stays None: with right padding, causal masking alone keeps every
    real query from attending pad keys, and pad rows' outputs are discarded.
    The pad k/v written past each prompt are stale until decode overwrites
    them; decode's mask hides them meanwhile.

    Shapes:
        slot_ids: [batch] int64 -- which pool slot each row writes to
    """

    attn_mask: torch.Tensor | None = None

    def __init__(self, manager: PreallocatedKVManager, slot_ids: torch.Tensor) -> None:
        self._manager = manager
        self._slot_ids = slot_ids

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the whole padded prompt block, pad positions included.

        Shapes:
            k, v:    [batch, seq_len, num_kv_heads, head_dim]
            -> k, v: same (nothing is cached yet beyond this block, so the
                     step reads back exactly what it wrote)
        """
        padded_len = k.shape[1]
        self._manager.k_cache[layer_idx][self._slot_ids, :padded_len] = k
        self._manager.v_cache[layer_idx][self._slot_ids, :padded_len] = v
        return k, v


class _DecodeStep:
    """Appends one token per sequence at each sequence's current length.

    Returned k/v are cache slices padded to the longest sequence in the step;
    attn_mask marks which positions are real per row, hiding both stale pad
    entries and shorter sequences' tails.

    Shapes:
        slot_ids:  [batch] int64 -- which pool slot each row reads and writes
        write_pos: [batch] int64 -- the position this step's token lands on,
                   i.e. each sequence's cached length before the step
        attn_mask: [batch, 1, 1, kv_len] bool
      where kv_len == max(write_pos) + 1
    """

    def __init__(
        self,
        manager: PreallocatedKVManager,
        slot_ids: torch.Tensor,
        write_pos: torch.Tensor,
        kv_len: int,
    ) -> None:
        self._manager = manager
        self._slot_ids = slot_ids
        self._write_pos = write_pos
        self._kv_len = kv_len
        keys_valid = torch.arange(kv_len, device=manager.device) <= write_pos[:, None]
        self.attn_mask: torch.Tensor | None = keys_valid.view(
            len(slot_ids), 1, 1, kv_len
        )

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's single token per row, then read the slots back.

        Shapes:
            k, v:    [batch, 1, num_kv_heads, head_dim]
            -> k, v: [batch, kv_len, num_kv_heads, head_dim]
          where kv_len is the longest sequence in the step, this token included
        """
        if k.shape[1] != 1:
            raise ValueError(f"decode appends one token per sequence, got {k.shape[1]}")
        k_cache = self._manager.k_cache[layer_idx]
        v_cache = self._manager.v_cache[layer_idx]
        k_cache[self._slot_ids, self._write_pos] = k[:, 0]
        v_cache[self._slot_ids, self._write_pos] = v[:, 0]
        return (
            k_cache[self._slot_ids, : self._kv_len],
            v_cache[self._slot_ids, : self._kv_len],
        )
