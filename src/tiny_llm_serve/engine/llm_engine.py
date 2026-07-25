import argparse
import sys

import torch

from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.layers.sampler import Sampler
from tiny_llm_serve.models import loader


class LLM:
    """Offline single-prompt generation over the naive KV cache."""

    def __init__(
        self, model: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        model_path = loader.resolve_model_path(model)
        self.tokenizer = loader.load_tokenizer(model_path)
        self.model = loader.load_model(model_path, device=device, dtype=dtype)
        self.sampler = Sampler()
        self.device = device

    @torch.inference_mode()
    def generate_ids(
        self, prompt: str | list[int], params: SamplingParams | None = None
    ) -> list[int]:
        """Decode the completion token ids for `prompt` (text or token ids).

        Accepting pre-tokenized prompts lets benchmarks keep tokenizer time out
        of engine measurements. Stops at the EOS token, once a `params.stop`
        substring appears in the decoded text, or after `params.max_tokens`,
        whichever comes first. A stop string can end mid-token, so the ids may
        overshoot it; `generate` trims exactly. `params=None` means greedy with
        the default budget.
        """
        if params is None:
            params = SamplingParams(temperature=0.0)
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        prompt_ids = (
            self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        )
        seen_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        kv_cache = self.model.new_kv_cache()
        positions = torch.arange(len(seen_ids), device=self.device)
        logits = self.model(seen_ids, positions, kv_cache)
        output_ids: list[int] = []
        for _ in range(params.max_tokens):
            next_id = self.sampler(
                logits[-1:], params, seen_ids.unsqueeze(0), generator
            )
            if not params.ignore_eos and next_id.item() == self.tokenizer.eos_token_id:
                break
            output_ids.append(next_id.item())
            seen_ids = torch.cat((seen_ids, next_id))
            if params.stop:
                # TODO: O(n^2) — re-decodes and re-scans all output_ids each
                # step. Replace with a streaming detokenizer (incremental
                # decode + tail scan) when we add token streaming.
                text = self.tokenizer.decode(output_ids)
                if any(s in text for s in params.stop):
                    break
            positions = torch.tensor([kv_cache.seq_len], device=self.device)
            logits = self.model(next_id, positions, kv_cache)
        return output_ids

    def generate(self, prompt: str, params: SamplingParams | None = None) -> str:
        """Decode a completion for `prompt` (raw text, no chat template).

        Stops at the EOS token, the first `params.stop` substring, or
        `params.max_tokens`, whichever comes first.
        """
        text = self.tokenizer.decode(self.generate_ids(prompt, params))
        stop = params.stop if params is not None else []
        cuts = [text.index(s) for s in stop if s in text]
        # decoding a flat id list yields a single str; the stub widens it to str | list[str]
        return text[: min(cuts)] if cuts else text  # pyrefly: ignore[bad-return]


def main() -> None:
    parser = argparse.ArgumentParser(prog="tiny-llm-serve")
    parser.add_argument("prompt", nargs="?", default="Where is Winterfell?")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = loader.resolve_device()
    print(f"device: {device}", file=sys.stderr)
    llm = LLM(args.model, device=device)
    params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
        stop=args.stop,
        seed=args.seed,
    )
    print(args.prompt + llm.generate(args.prompt, params))


if __name__ == "__main__":
    main()
