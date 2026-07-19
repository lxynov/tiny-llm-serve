import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rope_theta: float
    rms_norm_eps: float
    max_position_embeddings: int
    # 0 means "derive from hidden_size // num_attention_heads" in __post_init__
    head_dim: int = 0
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    architectures: list[str] = field(default_factory=list)
    quantization_config: dict | None = None

    def __post_init__(self) -> None:
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        data = json.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
