import torch
from backends import backends
from tinymodel import tiny_config, tiny_hf_config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from tiny_llm_serve.kv import PreallocatedKVManager
from tiny_llm_serve.models.qwen3 import Qwen3ForCausalLM


@backends("all")
def test_forward_returns_logits(device):
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (5,), device=device)

    logits = model(input_ids, torch.arange(5, device=device))

    assert logits.shape == (5, config.vocab_size)
    assert logits.isfinite().all()


def test_lm_head_tied_to_embeddings():
    model = Qwen3ForCausalLM(tiny_config())

    assert model.lm_head.weight is model.model.embed_tokens.weight


@backends("all")
def test_logits_match_hf(device):
    torch.manual_seed(0)
    ref = HFQwen3ForCausalLM(tiny_hf_config()).eval()
    model = Qwen3ForCausalLM(tiny_config()).eval()
    model.load_weights(ref.state_dict())
    ref = ref.to(device)
    model = model.to(device)
    input_ids = torch.randint(0, 128, (7,), device=device)

    with torch.no_grad():
        ref_logits = ref(input_ids.unsqueeze(0)).logits[0]
        logits = model(input_ids, torch.arange(7, device=device))

    torch.testing.assert_close(logits, ref_logits, atol=1e-4, rtol=1e-4)


@backends("all")
def test_batched_forward_matches_per_sequence(device):
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).eval().to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, 6), device=device)
    positions = torch.arange(6, device=device).expand(2, 6)

    with torch.no_grad():
        batched = model(input_ids, positions)
        singles = torch.stack([model(input_ids[i], positions[i]) for i in range(2)])

    torch.testing.assert_close(batched, singles, atol=1e-4, rtol=1e-4)


@backends("all")
def test_logits_indices_select_the_same_rows_as_a_full_forward(device):
    """Scoring only the wanted positions is exact, in both token layouts."""
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).eval().to(device)
    input_ids = torch.randint(0, config.vocab_size, (3, 6), device=device)
    positions = torch.arange(6, device=device).expand(3, 6)
    lens = torch.tensor([6, 4, 2], device=device)  # ragged: not all the last row

    with torch.no_grad():
        padded = model(input_ids, positions)
        padded_selected = model(input_ids, positions, logits_indices=lens - 1)
        flat = model(input_ids[0], positions[0])
        flat_selected = model(
            input_ids[0], positions[0], logits_indices=torch.tensor([-1], device=device)
        )

    rows = torch.arange(3, device=device)
    torch.testing.assert_close(
        padded_selected, padded[rows, lens - 1], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(flat_selected, flat[-1:], atol=1e-4, rtol=1e-4)


@backends("all")
def test_incremental_decode_matches_full_forward(device):
    """Prefilling a prompt and then stepping over the cache reproduces the
    logits of one forward pass over the whole sequence."""
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).eval().to(device)
    input_ids = torch.randint(0, config.vocab_size, (1, 6), device=device)
    positions = torch.arange(6, device=device).unsqueeze(0)

    with torch.no_grad():
        full_logits = model(input_ids, positions)

        manager = PreallocatedKVManager(
            config,
            num_slots=1,
            max_model_len=6,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        slot = manager.admit(3)
        prefill_logits = model(
            input_ids[:, :3], positions[:, :3], manager.begin_prefill([slot], [3])
        )
        step_logits = [prefill_logits[0, -1]]
        for pos in range(3, 6):
            logits = model(
                input_ids[:, pos : pos + 1],
                positions[:, pos : pos + 1],
                manager.begin_decode([slot]),
            )
            step_logits.append(logits[0, 0])

    assert int(manager.cached_seq_lens[slot]) == 6
    torch.testing.assert_close(
        torch.stack(step_logits), full_logits[0, 2:], atol=1e-4, rtol=1e-4
    )
