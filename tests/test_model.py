import torch
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.models.qwen3 import Qwen3ForCausalLM

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


def tiny_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )


def tiny_hf_config() -> Qwen3Config:
    ours = tiny_config()
    return Qwen3Config(
        hidden_size=ours.hidden_size,
        intermediate_size=ours.intermediate_size,
        num_hidden_layers=ours.num_hidden_layers,
        num_attention_heads=ours.num_attention_heads,
        num_key_value_heads=ours.num_key_value_heads,
        head_dim=ours.head_dim,
        vocab_size=ours.vocab_size,
        rope_parameters={"rope_type": "default", "rope_theta": ours.rope_theta},
        rms_norm_eps=ours.rms_norm_eps,
        max_position_embeddings=ours.max_position_embeddings,
        tie_word_embeddings=ours.tie_word_embeddings,
    )


def test_forward_returns_logits():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (5,))

    logits = model(input_ids, torch.arange(5))

    assert logits.shape == (5, config.vocab_size)
    assert logits.isfinite().all()


def test_lm_head_tied_to_embeddings():
    model = Qwen3ForCausalLM(tiny_config())

    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_logits_match_hf():
    torch.manual_seed(0)
    ref = HFQwen3ForCausalLM(tiny_hf_config()).eval()
    model = Qwen3ForCausalLM(tiny_config()).eval()
    model.load_weights(ref.state_dict())
    input_ids = torch.randint(0, 128, (7,))

    with torch.no_grad():
        ref_logits = ref(input_ids.unsqueeze(0)).logits[0]
        logits = model(input_ids, torch.arange(7))

    torch.testing.assert_close(logits, ref_logits, atol=1e-4, rtol=1e-4)


def test_incremental_decode_matches_full_forward():
    torch.manual_seed(0)
    config = tiny_config()
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (6,))

    with torch.no_grad():
        full_logits = model(input_ids, torch.arange(6))

        kv_cache = model.new_kv_cache()
        prefill_logits = model(input_ids[:3], torch.arange(3), kv_cache)
        step_logits = [prefill_logits[-1]]
        for pos in range(3, 6):
            logits = model(input_ids[pos : pos + 1], torch.tensor([pos]), kv_cache)
            step_logits.append(logits[0])

    assert kv_cache.seq_len == 6
    torch.testing.assert_close(
        torch.stack(step_logits), full_logits[2:], atol=1e-4, rtol=1e-4
    )
