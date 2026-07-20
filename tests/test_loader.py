import json

import pytest
import torch
from safetensors.torch import save_file

from tiny_llm_serve.models import loader
from tiny_llm_serve.models.qwen3 import Qwen3ForCausalLM


def test_load_weights_merges_shards(tmp_path):
    save_file(
        {"a.weight": torch.zeros(2, 3)}, tmp_path / "model-00001-of-00002.safetensors"
    )
    save_file(
        {"b.weight": torch.ones(4)}, tmp_path / "model-00002-of-00002.safetensors"
    )

    weights = loader.load_weights(tmp_path)

    assert set(weights) == {"a.weight", "b.weight"}
    assert weights["a.weight"].shape == (2, 3)
    assert torch.equal(weights["b.weight"], torch.ones(4))


def test_load_weights_requires_safetensors_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_weights(tmp_path)


def write_config(tmp_path, **overrides):
    data = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 100,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "architectures": ["Qwen3ForCausalLM"],
        **overrides,
    }
    (tmp_path / "config.json").write_text(json.dumps(data))


def test_config_ignores_unknown_keys_and_computes_head_dim(tmp_path):
    write_config(tmp_path, transformers_version="ignored")

    config = loader.load_config(tmp_path)

    assert config.head_dim == 16
    assert config.num_key_value_heads == 2
    assert config.architectures == ["Qwen3ForCausalLM"]


def test_resolve_model_class_supports_qwen3(tmp_path):
    write_config(tmp_path)

    model_class = loader.resolve_model_class(loader.load_config(tmp_path))

    assert model_class is Qwen3ForCausalLM


def test_load_model_rejects_unsupported_architecture(tmp_path):
    write_config(tmp_path, architectures=["Qwen3MoeForCausalLM"])

    with pytest.raises(ValueError, match="Qwen3MoeForCausalLM.*not supported"):
        loader.load_model(tmp_path)


def test_load_model_rejects_quantized_checkpoint(tmp_path):
    write_config(tmp_path, quantization_config={"quant_method": "fp8"})

    with pytest.raises(ValueError, match="quantized.*fp8.*not supported"):
        loader.load_model(tmp_path)


def test_qwen3_config(qwen3_path):
    config = loader.load_config(qwen3_path)

    assert config.num_hidden_layers > 0
    assert config.num_attention_heads % config.num_key_value_heads == 0
    assert config.vocab_size > 0
    assert config.rope_theta == 1000000.0


def test_qwen3_tokenizer_round_trips(qwen3_path):
    tokenizer = loader.load_tokenizer(qwen3_path)

    text = "Lin is Forest."
    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text


def test_qwen3_chat_template(qwen3_path):
    tokenizer = loader.load_tokenizer(qwen3_path)

    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}], add_generation_prompt=True
    )
    text = tokenizer.decode(token_ids)  # type: ignore

    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text


def test_qwen3_weight_names_and_shapes(qwen3_path):
    config = loader.load_config(qwen3_path)
    weights = loader.load_weights(qwen3_path)

    assert weights["model.embed_tokens.weight"].shape == (
        config.vocab_size,
        config.hidden_size,
    )
    assert "model.norm.weight" in weights
    per_layer_names = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    )
    for layer in (0, config.num_hidden_layers - 1):
        for name in per_layer_names:
            assert f"model.layers.{layer}.{name}" in weights

    q_proj = weights["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.shape == (
        config.num_attention_heads * config.head_dim,
        config.hidden_size,
    )
