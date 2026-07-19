from tiny_llm_serve.models.base import CausalLM
from tiny_llm_serve.models.qwen3 import Qwen3ForCausalLM

# HF config.json `architectures` entry -> model class. Covers the dense Qwen3
# family (0.6B-32B, including the 2507 refreshes); MoE (Qwen3Moe*, Qwen3Next*)
# and multimodal variants would need their own implementations.
MODEL_REGISTRY: dict[str, type[CausalLM]] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}
