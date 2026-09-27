import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from tiny_llm_serve.engine.sampling_params import SamplingParams
from tiny_llm_serve.kv import PreallocatedKVManager
from tiny_llm_serve.layers.sampler import Sampler
from tiny_llm_serve.models import loader


@dataclass
class Timing:
    prefill_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0


@dataclass
class DecodeState:
    """One static batch mid-flight: everything a decode step reads and advances.

    `step_logits` is a step's input, left by the prefill or the previous step
    for this step to sample from. Every other tensor is allocated once and
    written in place, so shapes never change (torch.compile, CUDA graphs).
    Finishing is decided on the device; a step reads back only whether all rows
    are done.

    Shapes:
        step_logits: [batch, vocab_size]
        lens:        [batch] int64 -- tokens cached, i.e. the next token's position
        seen_ids:    [batch, prompt_width + budget] int64 -- for the repetition
                     penalty: prompts padded with their first token, then draws
        sampled:     [batch, budget] int64 -- view of seen_ids' tail
        first_ids:   [batch] int64 -- fed to finished rows while they idle
        max_tokens:  [batch] int64 -- per-row budget; `budget` is their max
        out_lens:    [batch] int64 -- ids each row keeps
        finished:    [batch] bool
    """

    manager: PreallocatedKVManager
    slots: list[int]
    params: SamplingParams
    eos: int | None
    detokenize: Callable[[list[int]], str]
    generator: torch.Generator | None
    step_logits: torch.Tensor
    lens: torch.Tensor
    seen_ids: torch.Tensor
    sampled: torch.Tensor
    first_ids: torch.Tensor
    max_tokens: torch.Tensor
    out_lens: torch.Tensor
    finished: torch.Tensor
    steps: int = 0
    done: bool = False

    def record(self, next_ids: torch.Tensor) -> None:
        """Store this step's draws and finish the rows they end.

        Shapes:
            next_ids: [batch] int64
        """
        step = self.steps  # next_ids holds each row's (step + 1)-th id
        # Finished rows' draws land past their out_len and are never read back.
        self.sampled[:, step] = next_ids
        if self.eos is not None and not self.params.ignore_eos:
            at_eos = ~self.finished & (next_ids == self.eos)
            self.out_lens.masked_fill_(at_eos, step)  # EOS is not an output id
            self.finished |= at_eos
        if self.params.stop:
            self._apply_stop_strings(step)
        self.finished |= self.max_tokens <= step + 1
        self.done = bool(self.finished.all())  # the step's one read back

    def outputs(self) -> list[list[int]]:
        """The ids each row kept, read back once the batch has drained."""
        return [
            ids[:n] for ids, n in zip(self.sampled.tolist(), self.out_lens.tolist())
        ]

    def _apply_stop_strings(self, step: int) -> None:
        """Finish every running row whose output contains a stop string.

        Checked on the host, so it costs a sync per step when `params.stop` is set.
        """
        # TODO: O(n^2) -- re-decodes each row's whole output every step. Switch
        # to a streaming detokenizer when we add token streaming.
        texts = [self.detokenize(ids) for ids in self.sampled[:, : step + 1].tolist()]
        hit = [any(s in text for s in self.params.stop) for text in texts]
        stopped = torch.tensor(hit, device=self.finished.device) & ~self.finished
        self.out_lens.masked_fill_(stopped, step + 1)
        self.finished |= stopped


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

    def generate(self, prompt: str, params: SamplingParams | None = None) -> str:
        """Decode a completion for `prompt` (raw text, no chat template).

        Stops at the EOS token, the first `params.stop` substring, or
        `params.max_tokens`, whichever comes first.
        """
        return self.generate_batch([prompt], params)[0]

    @torch.inference_mode()
    def prefill_batch(
        self,
        prompts: list[list[int]],
        params: SamplingParams | list[SamplingParams] | None = None,
        max_model_len: int | None = None,
    ) -> DecodeState:
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
        manager = PreallocatedKVManager(
            self.model.config,
            num_slots=len(prompts),
            max_model_len=max_model_len,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        slots = [manager.admit(n) for n in prompt_lens]

        num_seqs, padded_len = len(prompts), max(prompt_lens)
        budget = max(p.max_tokens for p in params_list)
        input_ids = torch.zeros((num_seqs, padded_len), dtype=torch.long)
        # Sized up front so that its shape stays fixed, which is needed by
        # torch.compile and CUDA graphs.
        seen_ids = torch.zeros((num_seqs, padded_len + budget), dtype=torch.long)
        for i, prompt in enumerate(prompts):
            ids = torch.tensor(prompt, dtype=torch.long)
            input_ids[i, : len(prompt)] = ids
            seen_ids[i] = prompt[0]
            seen_ids[i, : len(prompt)] = ids
        input_ids = input_ids.to(self.device)
        seen_ids = seen_ids.to(self.device)
        positions = torch.arange(padded_len, device=self.device).expand(num_seqs, -1)
        lens = torch.tensor(prompt_lens, device=self.device)
        max_tokens = torch.tensor(
            [p.max_tokens for p in params_list], device=self.device
        )
        step_logits = self.model(
            input_ids, positions, manager.begin_prefill(slots, prompt_lens), lens - 1
        )

        return DecodeState(
            manager=manager,
            slots=slots,
            params=sampling_params,
            eos=self.tokenizer.eos_token_id,
            # decoding a flat id list yields a single str; the stub widens it to str | list[str]
            detokenize=self.tokenizer.decode,  # pyrefly: ignore[bad-argument-type]
            generator=generator,
            step_logits=step_logits,
            lens=lens,
            seen_ids=seen_ids,
            sampled=seen_ids[:, padded_len:],
            first_ids=seen_ids[:, 0],
            max_tokens=max_tokens,
            out_lens=max_tokens.clone(),
            finished=torch.zeros(num_seqs, dtype=torch.bool, device=self.device),
        )

    @torch.inference_mode()
    def decode_step(self, state: DecodeState) -> torch.Tensor:
        """Advance a static batch by one token, and return what each row drew.
        Shapes:
            -> [batch] int64
        """
        next_ids = self.sampler(
            state.step_logits, state.params, state.seen_ids, state.generator
        )
        state.record(next_ids)
        if state.done:
            return next_ids
        # Finished rows idle: feed their (already seen) first token so the
        # step stays well-formed for any vocab, and discard their output.
        feed = torch.where(state.finished, state.first_ids, next_ids)
        logits = self.model(
            feed.unsqueeze(1),
            state.lens.unsqueeze(1),
            state.manager.begin_decode(state.slots),
        )
        state.step_logits = logits[:, 0]
        state.lens += 1
        state.steps += 1
        return next_ids

    def generate_batch_ids(
        self,
        prompts: list[list[int]],
        params: SamplingParams | list[SamplingParams] | None = None,
        max_model_len: int | None = None,
        timing: Timing | None = None,
    ) -> list[list[int]]:
        """Decode a static batch: one padded prefill, then lockstep decode."""
        start = self._now() if timing is not None else 0.0
        state = self.prefill_batch(prompts, params, max_model_len)
        prefill_end = self._now() if timing is not None else 0.0
        if timing is not None:
            timing.prefill_s += prefill_end - start
        while not state.done:
            self.decode_step(state)
        if timing is not None:
            timing.decode_s += self._now() - prefill_end
            timing.decode_steps += state.steps
        return state.outputs()

    def generate_batch(
        self,
        prompts: list[str],
        params: SamplingParams | list[SamplingParams] | None = None,
    ) -> list[str]:
        """Decode completions for `prompts` together (raw text, no chat
        template). See `generate_batch_ids` for the batching semantics."""
        params_list = _batch_params(params, len(prompts))
        prompt_ids = [self.tokenizer.encode(p) for p in prompts]
        outputs = self.generate_batch_ids(prompt_ids, params_list)
        stop = params_list[0].stop  # rows differ only in max_tokens
        # decoding a flat id list yields a single str; the stub widens it to str | list[str]
        return [_trim(self.tokenizer.decode(o), stop) for o in outputs]  # pyrefly: ignore[bad-argument-type]


def _trim(text: str, stop: list[str]) -> str:
    """Cut `text` at the first stop substring. A stop string can end mid-token,
    so the ids a row stopped on may run past it."""
    cuts = [text.index(s) for s in stop if s in text]
    return text[: min(cuts)] if cuts else text


def _batch_params(
    params: SamplingParams | list[SamplingParams] | None, num_seqs: int
) -> list[SamplingParams]:
    """Normalize to one SamplingParams per sequence and reject what a single
    batched sampler pass cannot honor: per-row sampling controls, of which only
    max_tokens may vary."""
    if params is None:
        params = SamplingParams(temperature=0.0)
    if isinstance(params, SamplingParams):
        params = [params] * num_seqs
    if len(params) != num_seqs:
        raise ValueError(f"got {len(params)} sampling params for {num_seqs} prompts")
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
