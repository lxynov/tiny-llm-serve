import torch
from torch import nn

from tiny_llm_serve.engine.sampling_params import SamplingParams


class Sampler(nn.Module):
    """Sampling pipeline: repetition penalty -> temperature -> top-k -> top-p ->
    multinomial. All filtering happens on fp32 copies; the input logits are
    never mutated."""

    def forward(
        self,
        logits: torch.Tensor,
        params: SamplingParams | None = None,
        seen_token_ids: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one next token id per sequence.

        `seen_token_ids` holds the prompt+output ids the repetition penalty
        applies to. `generator` makes draws reproducible. `params=None` means
        greedy.

        Shapes:
            logits:         [num_seqs, vocab_size]
            seen_token_ids: [num_seqs, num_seen] int64 | None
            ->              [num_seqs] int64
        """
        if params is None:
            params = SamplingParams(temperature=0.0)
        logits = logits.float()
        if params.repetition_penalty != 1.0 and seen_token_ids is not None:
            logits = _apply_repetition_penalty(
                logits, seen_token_ids, params.repetition_penalty
            )
        if params.temperature == 0.0:
            return logits.argmax(dim=-1)
        logits = logits / params.temperature
        if params.top_k != -1:
            logits = _apply_top_k(logits, params.top_k)
        if params.top_p < 1.0:
            logits = _apply_top_p(logits, params.top_p)
        # softmax subtracts the row max internally, so scaled-up logits are safe
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _apply_repetition_penalty(
    logits: torch.Tensor, token_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Discourage seen tokens: divide their positive logits by `penalty`,
    multiply their negative logits by it.

    Shapes:
        logits:    [num_seqs, vocab_size]
        token_ids: [num_seqs, num_seen] int64
        ->         same as logits
    """
    scores = logits.gather(1, token_ids)
    scores = torch.where(scores > 0, scores / penalty, scores * penalty)
    return logits.scatter(1, token_ids, scores)


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask all but the k highest logits per row (ties at the boundary survive).

    Shapes: logits: [num_seqs, vocab_size] -> same
    """
    k = min(k, logits.shape[-1])
    kth = logits.topk(k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest high-probability prefix whose
    cumulative probability reaches p; the token crossing the threshold stays.

    Shapes: logits: [num_seqs, vocab_size] -> same
    """
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    drop = cumulative > p
    drop[:, 1:] = drop[:, :-1].clone()  # shift right: keep the crossing token
    drop[:, 0] = False
    return logits.masked_fill(drop.scatter(1, sorted_indices, drop), float("-inf"))
