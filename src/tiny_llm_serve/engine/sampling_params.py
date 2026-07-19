from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request sampling controls.

    Applied in order: repetition penalty -> temperature -> top-k -> top-p ->
    multinomial draw. temperature 0 means greedy (argmax after penalties).
    """

    temperature: float = 1.0
    top_k: int = -1  # -1 disables; otherwise sample among the k highest logits
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 128
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
