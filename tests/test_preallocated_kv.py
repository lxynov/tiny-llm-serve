import pytest
import torch
from tinymodel import HEAD_DIM, NUM_KV_HEADS, tiny_config

from tiny_llm_serve.kv import PreallocatedKVManager


def make_manager(
    num_slots=2, max_model_len=8, kv_len_bucket=1
) -> PreallocatedKVManager:
    """A one-token bucket by default, so a test that is about slots and lengths
    reads in them; the bucketing tests below set their own."""
    config = tiny_config(num_hidden_layers=1)
    return PreallocatedKVManager(
        config,
        num_slots,
        max_model_len,
        device="cpu",
        dtype=torch.float32,
        kv_len_bucket=kv_len_bucket,
    )


def kv(batch: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (batch, length, NUM_KV_HEADS, HEAD_DIM)
    return torch.randn(shape), torch.randn(shape)


def test_admission_exhausts_slots():
    manager = make_manager(num_slots=2)

    assert manager.can_admit(4)
    first, second = manager.admit(4), manager.admit(4)

    assert {first, second} == {0, 1}
    assert not manager.can_admit(4)
    with pytest.raises(ValueError, match="cannot admit"):
        manager.admit(4)


def test_admission_rejects_prompts_larger_than_a_slot():
    manager = make_manager(max_model_len=8)

    assert not manager.can_admit(9)
    with pytest.raises(ValueError, match="cannot admit"):
        manager.admit(9)


def test_free_recycles_slot_and_rejects_double_free():
    manager = make_manager(num_slots=1)
    slot = manager.admit(4)
    manager.cached_seq_lens[slot] = 4  # as if prefilled

    manager.free(slot)

    assert manager.can_admit(4)
    assert manager.cached_seq_lens[slot] == 0
    with pytest.raises(ValueError, match="already free"):
        manager.free(slot)


def test_prefill_writes_each_prompt_into_its_slot():
    torch.manual_seed(0)
    manager = make_manager()
    slots = [manager.admit(2), manager.admit(3)]
    k, v = kv(2, 3)  # right-padded to the longest prompt

    step = manager.begin_prefill(slots, [2, 3])
    k_out, v_out = step.append(0, k, v)

    assert step.attn_mask is None
    assert k_out is k and v_out is v  # prefill attends the padded block as-is
    assert torch.equal(manager.k_cache[0][slots[0], :3], k[0])
    assert torch.equal(manager.v_cache[0][slots[1], :3], v[1])
    assert manager.cached_seq_lens[slots].tolist() == [2, 3]


def test_prefill_requires_fresh_slots():
    manager = make_manager()
    slots = [manager.admit(2)]
    manager.begin_prefill(slots, [2])

    with pytest.raises(ValueError, match="occupied"):
        manager.begin_prefill(slots, [2])


def test_decode_appends_at_each_sequence_length():
    torch.manual_seed(0)
    manager = make_manager()
    slots = [manager.admit(2), manager.admit(3)]
    manager.begin_prefill(slots, [2, 3]).append(0, *kv(2, 3))
    k_new, v_new = kv(2, 1)

    step = manager.begin_decode(slots)
    k_out, _ = step.append(0, k_new, v_new)

    # New tokens land at positions 2 and 3; the padded view spans max+1 = 4.
    assert torch.equal(manager.k_cache[0][slots[0], 2], k_new[0, 0])
    assert torch.equal(manager.k_cache[0][slots[1], 3], k_new[1, 0])
    assert k_out.shape == (2, 4, NUM_KV_HEADS, HEAD_DIM)
    assert step.attn_mask is not None
    assert step.attn_mask.shape == (2, 1, 1, 4)
    assert step.attn_mask.flatten(0, 2).tolist() == [
        [True, True, True, False],  # row 0 is valid through its new token at 2
        [True, True, True, True],
    ]
    assert manager.cached_seq_lens[slots].tolist() == [3, 4]


def test_decode_rejects_multi_token_input():
    manager = make_manager()
    slots = [manager.admit(2)]
    manager.begin_prefill(slots, [2]).append(0, *kv(1, 2))

    with pytest.raises(ValueError, match="one token"):
        manager.begin_decode(slots).append(0, *kv(1, 2))


def test_decode_raises_when_a_slot_is_full():
    manager = make_manager(max_model_len=4)
    slots = [manager.admit(4)]
    manager.begin_prefill(slots, [4]).append(0, *kv(1, 4))

    with pytest.raises(RuntimeError, match="out of KV capacity"):
        manager.begin_decode(slots)


def test_decode_requires_prefilled_slots():
    manager = make_manager()
    slots = [manager.admit(2)]

    with pytest.raises(ValueError, match="unprefilled"):
        manager.begin_decode(slots)


def test_decode_rounds_kv_len_up_to_a_bucket():
    """Decode hands back a bucketed window so a run traces a handful of shapes
    instead of one per step. The positions past the sequence are masked, so the
    bucket costs bytes and not correctness."""
    manager = make_manager(num_slots=1, max_model_len=16, kv_len_bucket=4)
    slots = [manager.admit(2)]
    manager.begin_prefill(slots, [2]).append(0, *kv(1, 2))

    step = manager.begin_decode(slots)
    k_out, _ = step.append(0, *kv(1, 1))

    assert k_out.shape[1] == 4  # 3 cached tokens, rounded up to the bucket
    assert step.attn_mask is not None
    assert step.attn_mask.flatten().tolist() == [True, True, True, False]


def test_the_bucket_never_reads_past_a_slot():
    """A slot is the hard edge: the last bucket of a full slot is short."""
    manager = make_manager(num_slots=1, max_model_len=6, kv_len_bucket=4)
    slots = [manager.admit(4)]
    manager.begin_prefill(slots, [4]).append(0, *kv(1, 4))

    k_out, _ = manager.begin_decode(slots).append(0, *kv(1, 1))

    assert k_out.shape[1] == 6  # 5 cached, bucketed to 8, capped by the slot
