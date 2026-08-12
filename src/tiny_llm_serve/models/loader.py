import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.models import MODEL_REGISTRY
from tiny_llm_serve.models.base import CausalLM


def resolve_model_path(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model)
    return Path(snapshot_download(model))


def resolve_device(device: str | None = None) -> str:
    """Return `device` if given, else the best available: cuda, then mps, then cpu."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    """Block until `device` has finished the work already submitted to it.

    Accelerator work is enqueued asynchronously, so a clock read that is not
    preceded by a barrier times kernel *submission* rather than execution.
    """
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def load_config(model_path: Path) -> ModelConfig:
    return ModelConfig.from_json(model_path / "config.json")


def load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    # every backend it can return subclasses PreTrainedTokenizerBase; only the
    # `| None` in its signature (for paths it can't resolve) widens the type
    return AutoTokenizer.from_pretrained(model_path)  # pyrefly: ignore[bad-return]


def resolve_model_class(config: ModelConfig) -> type[CausalLM]:
    for arch in config.architectures:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(
        f"model architectures {config.architectures} are not supported; "
        f"supported architectures: {sorted(MODEL_REGISTRY)}"
    )


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


def load_model(
    model_path: Path, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> CausalLM:
    """Build the model on `device` with `dtype` params and load checkpoint weights.

    We set the default dtype before initialization rather than casting the model
    afterwards (i.e., `model.to(dtype)`). A blanket `.to(dtype)` cast would recursively
    convert all parameters and buffers. By changing the default dtype instead, we preserve
    the precision of buffers that explicitly request float32 during initialization
    (such as the RoPE cos/sin frequency caches), which are sensitive to numerical precision.
    """
    config = load_config(model_path)
    model_class = resolve_model_class(config)
    if config.quantization_config is not None:
        # e.g. the FP8 Qwen3 checkpoints: same architecture name, but their weights
        # are stored quantized with per-block scales, which load_weights can't handle
        quant_method = config.quantization_config.get("quant_method", "unknown")
        raise ValueError(f"quantized checkpoints ({quant_method}) are not supported")
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = model_class(config)
    finally:
        torch.set_default_dtype(default_dtype)
    model.load_weights(load_weights(model_path))
    return model.eval()


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
