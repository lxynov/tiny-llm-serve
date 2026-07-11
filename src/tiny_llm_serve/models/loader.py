import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tiny_llm_serve.config import ModelConfig


def resolve_model_path(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    return Path(snapshot_download(model))


def load_config(model_path: Path) -> ModelConfig:
    return ModelConfig.from_json(model_path / "config.json")


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_path)


def load_weights(model_path: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards under `model_path` into a name -> tensor dict."""
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {model_path}")
    weights: dict[str, torch.Tensor] = {}
    for file in files:
        with safe_open(file, framework="pt") as f:
            for name in f.keys():
                weights[name] = f.get_tensor(name)
    return weights


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    model_path = resolve_model_path(model)
    print(load_config(model_path))
    weights = load_weights(model_path)
    for name, tensor in weights.items():
        print(f"{name:<55} {str(tuple(tensor.shape)):<15} {tensor.dtype}")
    num_params = sum(t.numel() for t in weights.values())
    print(f"{len(weights)} tensors, {num_params / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
