"""A tiny random-weight Qwen3 configuration shared across test modules.

Small enough for CPU tests, structurally identical to the real model (GQA,
tied embeddings), with an HF twin for parity checks and checkpoint writing.
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from tiny_llm_serve.config import ModelConfig

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8


def tiny_config(**overrides) -> ModelConfig:
    defaults = dict(
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
    return ModelConfig(**{**defaults, **overrides})


def tiny_hf_config(config: ModelConfig | None = None):
    from transformers import Qwen3Config

    ours = config if config is not None else tiny_config()
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


def save_checkpoint(target: Path, tokenizer_path: Path, **overrides) -> Path:
    """Write a loadable checkpoint: tiny random weights, a real tokenizer.

    Weights are saved in HF layout (split q/k/v projections) so loading
    exercises the same shard-merging path as a hub checkpoint. The tokenizer
    files are copied from a real snapshot because the loader always needs one;
    its vocab is far larger than the tiny model's, which is harmless for
    tests that only feed ids below the tiny vocab size.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    config = tiny_config(architectures=["Qwen3ForCausalLM"], **overrides)
    (target / "config.json").write_text(json.dumps(asdict(config)))
    torch.manual_seed(0)
    state = dict(HFQwen3ForCausalLM(tiny_hf_config(config)).state_dict())
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)  # safetensors rejects shared tensors
    save_file(state, str(target / "model.safetensors"))
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = tokenizer_path / name
        if source.exists():
            shutil.copy(source, target / name)
    return target
