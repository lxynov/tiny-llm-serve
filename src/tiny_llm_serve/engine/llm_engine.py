import argparse
import sys
import time
from dataclasses import dataclass, replace

import torch

from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.kv import NaiveKVCache, PreallocatedKVManager
from tiny_llm_serve.layers.sampler import Sampler
from tiny_llm_serve.models import loader


@dataclass
class Timing:
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0


class LLM:
    def __init__(
        self, model: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        model_path = loader.resolve_model_path(model)
        self.tokenizer = loader.load_tokenizer(model_path)
        self.model = loader.load_model(model_path, device=device, dtype=dtype)
        self.sampler = Sampler()
        self.device = device

    def _now(self) -> float:
        """A clock read the device has actually caught up to."""
        loader.synchronize(self.device)
        return time.perf_counter()

    @torch.inference_mode()
    def generate_ids(
        self,
        prompt: str | list[int],
        params: SamplingParams | None = None,
        timing: Timing | None = None,
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
        start = prefill_end = self._now() if timing is not None else 0.0
        seen_ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        kv_cache = NaiveKVCache(self.model.config.num_hidden_layers)
        positions = torch.arange(len(seen_ids), device=self.device)
        step_logits = self.model(
            seen_ids, positions, kv_cache, torch.tensor([-1], device=self.device)
        )
        if timing is not None:
            prefill_end = self._now()
            timing.prefill_s += prefill_end - start
        output_ids: list[int] = []
        for _ in range(params.max_tokens):
            next_id = self.sampler(
                step_logits, params, seen_ids.unsqueeze(0), generator
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
            step_logits = self.model(next_id, positions, kv_cache)
            if timing is not None:
                timing.decode_steps += 1
        if timing is not None:
            timing.decode_s += self._now() - prefill_end
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

    @torch.inference_mode()
    def generate_batch_ids(
        self,
        prompts: list[list[int]],
        params: SamplingParams | list[SamplingParams] | None = None,
        max_model_len: int | None = None,
        timing: Timing | None = None,
    ) -> list[list[int]]:
        """Decode a static batch: one padded prefill, then lockstep decode.

        A sequence that hits EOS or its max_tokens idles in its slot (still
        stepped, output discarded) until the whole batch finishes; nothing new
        is admitted mid-flight. `params` may be a list to vary max_tokens per
        sequence. Each preallocated slot reserves `max_model_len` tokens
        (default: sized so no sequence can outgrow it). Greedy batched output
        matches sequential exactly; seeded sampling draws in batch order, so it
        reproduces itself but not sequential runs.
        """
        params_list = _batch_params(params, len(prompts))
        sampling_params = params_list[0]  # rows differ only in max_tokens
        generator = None
        if sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(
                sampling_params.seed
            )
        prompt_lens = [len(p) for p in prompts]
        if min(prompt_lens, default=0) == 0:
            raise ValueError("empty prompts are not supported")
        if max_model_len is None:
            # An early finisher idles (and keeps writing KV) until the whole
            # batch drains, so every slot must fit the longest prompt plus the
            # largest budget, not just its own sequence.
            max_model_len = max(prompt_lens) + max(p.max_tokens for p in params_list)
        start = prefill_end = self._now() if timing is not None else 0.0
        manager = PreallocatedKVManager(
            self.model.config,
            num_slots=len(prompts),
            max_model_len=max_model_len,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        slots = [manager.admit(n) for n in prompt_lens]

        num_seqs, padded_len = len(prompts), max(prompt_lens)
        input_ids = torch.zeros((num_seqs, padded_len), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            input_ids[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        input_ids = input_ids.to(self.device)
        positions = torch.arange(padded_len, device=self.device).expand(num_seqs, -1)
        lens = torch.tensor(prompt_lens, device=self.device)
        step_logits = self.model(
            input_ids, positions, manager.begin_prefill(slots, prompt_lens), lens - 1
        )
        if timing is not None:
            prefill_end = self._now()
            timing.prefill_s += prefill_end - start

        # Track seen ids for the repetition penalty, padding each row with its
        # own first token: duplicate ids are harmless to the penalty (the same
        # penalized value is scattered twice), unlike an arbitrary pad id,
        # which would spuriously penalize a token the row never produced.
        seen_ids = input_ids.clone()
        for i, n in enumerate(prompt_lens):
            seen_ids[i, n:] = seen_ids[i, 0]

        eos = self.tokenizer.eos_token_id
        outputs: list[list[int]] = [[] for _ in prompts]
        finished = torch.zeros(num_seqs, dtype=torch.bool, device=self.device)
        while True:
            next_ids = self.sampler(step_logits, sampling_params, seen_ids, generator)
            for i, next_id in enumerate(next_ids.tolist()):
                if finished[i]:
                    continue
                if not sampling_params.ignore_eos and next_id == eos:
                    finished[i] = True
                elif len(outputs[i]) + 1 == params_list[i].max_tokens:
                    outputs[i].append(next_id)
                    finished[i] = True
                else:
                    outputs[i].append(next_id)
            if bool(finished.all()):
                break
            # Finished rows idle: feed their (already seen) first token so the
            # step stays well-formed for any vocab, and discard their output.
            feed = torch.where(finished, seen_ids[:, 0], next_ids)
            seen_ids = torch.cat((seen_ids, feed.unsqueeze(1)), dim=1)
            logits = self.model(
                feed.unsqueeze(1), lens.unsqueeze(1), manager.begin_decode(slots)
            )
            step_logits = logits[:, 0]
            lens = lens + 1
            if timing is not None:
                timing.decode_steps += 1
        if timing is not None:
            timing.decode_s += self._now() - prefill_end
        return outputs

    def generate_batch(
        self,
        prompts: list[str],
        params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """Decode completions for `prompts` together (raw text, no chat
        template). See `generate_batch_ids` for the batching semantics."""
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        outputs = self.generate_batch_ids(prompt_ids, params)
        # decoding a flat id list yields a single str; the stub widens it to str | list[str]
        return [self.tokenizer.decode(o) for o in outputs]  # pyrefly: ignore[bad-return]


def _batch_params(
    params: SamplingParams | list[SamplingParams] | None, num_seqs: int
) -> list[SamplingParams]:
    """Normalize to one SamplingParams per sequence and reject what a single
    batched sampler pass cannot honor: per-row sampling controls (only
    max_tokens may vary) and stop strings (no incremental detokenization)."""
    if params is None:
        params = SamplingParams(temperature=0.0)
    if isinstance(params, SamplingParams):
        params = [params] * num_seqs
    if len(params) != num_seqs:
        raise ValueError(f"got {len(params)} sampling params for {num_seqs} prompts")
    if any(p.stop for p in params):
        raise ValueError("stop strings are not supported in batched generation")
    first = params[0]
    if any(replace(p, max_tokens=first.max_tokens) != first for p in params[1:]):
        raise ValueError("batched sampling params may differ only in max_tokens")
    return params


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
