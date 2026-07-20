import sys

import torch

from tiny_llm_serve.models import loader


class LLM:
    """Offline single-prompt greedy generation over the naive KV cache."""

    def __init__(
        self, model: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        model_path = loader.resolve_model_path(model)
        self.tokenizer = loader.load_tokenizer(model_path)
        self.model = loader.load_model(model_path, device=device, dtype=dtype)
        self.device = device

    @torch.inference_mode()
    def generate_ids(
        self, prompt: str, max_tokens: int = 128, stop: list[str] | None = None
    ) -> list[int]:
        """Greedily decode the completion token ids for `prompt`.

        Stops at the EOS token, once a `stop` substring appears in the decoded
        text, or after `max_tokens`, whichever comes first. A stop string can
        end mid-token, so the ids may overshoot it; `generate` trims exactly.
        """
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.long, device=self.device
        )
        kv_cache = self.model.new_kv_cache()
        positions = torch.arange(len(input_ids), device=self.device)
        logits = self.model(input_ids, positions, kv_cache)
        output_ids: list[int] = []
        for _ in range(max_tokens):
            next_id = logits[-1:].argmax(dim=-1)
            if next_id.item() == self.tokenizer.eos_token_id:
                break
            output_ids.append(next_id.item())
            if stop:
                text = self.tokenizer.decode(output_ids)
                if any(s in text for s in stop):
                    break
            positions = torch.tensor([kv_cache.seq_len], device=self.device)
            logits = self.model(next_id, positions, kv_cache)
        return output_ids

    def generate(
        self, prompt: str, max_tokens: int = 128, stop: list[str] | None = None
    ) -> str:
        """Greedily decode a completion for `prompt` (raw text, no chat template).

        Stops at the EOS token, the first `stop` substring, or `max_tokens`,
        whichever comes first.
        """
        text = self.tokenizer.decode(self.generate_ids(prompt, max_tokens, stop))
        cuts = [text.index(s) for s in (stop or []) if s in text]
        # decoding a flat id list yields a single str; the stub widens it to str | list[str]
        return text[: min(cuts)] if cuts else text  # pyrefly: ignore[bad-return]


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Where is Winterfell?"
    device = loader.resolve_device()
    print(f"device: {device}", file=sys.stderr)
    llm = LLM(model, device=device)
    print(prompt + llm.generate(prompt, max_tokens=32))


if __name__ == "__main__":
    main()
