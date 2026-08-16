import torch
from torch import nn

from tiny_llm_serve.config import ModelConfig
from tiny_llm_serve.kv import KVCacheView
from tiny_llm_serve.layers.activation import SiluAndMul
from tiny_llm_serve.layers.attention import Attention
from tiny_llm_serve.layers.layernorm import RMSNorm
from tiny_llm_serve.layers.linear import MergedLinear
from tiny_llm_serve.layers.rotary_embedding import RotaryEmbedding
from tiny_llm_serve.models.base import CausalLM

# Checkpoint projection name -> (merged module attribute, shard index)
SHARD_MAP = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


class Qwen3Attention(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = MergedLinear(config.hidden_size, [q_size, kv_size, kv_size])
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        # Qwen3 normalizes q and k per head, before RoPE
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb
        self.attn = Attention()

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, kv_cache: KVCacheView | None
    ) -> torch.Tensor:
        """Project to q/k/v, normalize and rotate them, attend, project back.

        Shapes:
            x:                  [*b, seq_len, hidden_size]
            positions:          [*b, seq_len] int64
            qkv_proj(x):        [*b, seq_len, (num_heads + 2 * num_kv_heads) * head_dim]
            q, after split:     [*b, seq_len, num_heads * head_dim]
            k, v, after split:  [*b, seq_len, num_kv_heads * head_dim]
            q, after view:      [*b, seq_len, num_heads, head_dim]
            k, v, after view:   [*b, seq_len, num_kv_heads, head_dim]
            k, v, from cache:   [*b, kv_len, num_kv_heads, head_dim]
            attn(q, k, v):      [*b, seq_len, num_heads, head_dim]
            o.flatten(-2):      [*b, seq_len, num_heads * head_dim]
            ->                  same as x
          where kv_len >= seq_len, the positions the cache hands back this step;
          q_norm, k_norm and rotary_emb leave the per-head shape untouched

        Qwen3-0.6B: 16 heads, 8 kv heads (GQA 2:1), head_dim 128 from config
        rather than hidden_size // num_heads, so qkv_proj widens 1024 -> 4096
        and o_proj narrows 2048 -> 1024.
        """
        q, k, v = self.qkv_proj(x).split(self.qkv_proj.output_sizes, dim=-1)
        q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim))
        k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim))
        v = v.view(*v.shape[:-1], self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_mask = None
        if kv_cache is not None:
            k, v = kv_cache.append(self.layer_idx, k, v)
            attn_mask = kv_cache.attn_mask
        o = self.attn(q, k, v, attn_mask)
        return self.o_proj(o.flatten(-2))


class Qwen3MLP(nn.Module):
    """SwiGLU feed-forward network (Swish-Gated Linear Unit).

    FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down
    silu(z) = z * sigmoid(z)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedLinear(
            config.hidden_size, [config.intermediate_size] * 2
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project up to a fused [gate, up], gate it, then project back down.

        Shapes:
            x:               [..., hidden_size]
            gate_up_proj(x): [..., 2 * intermediate_size]
            act_fn(...):     [..., intermediate_size]
            ->               same as x
        """
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self, config: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, rotary_emb)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, kv_cache: KVCacheView | None
    ) -> torch.Tensor:
        """Pre-normed attention and MLP sublayers, each behind a residual.

        Shapes:
            x:         [*b, seq_len, hidden_size]
            positions: [*b, seq_len] int64
            ->         same as x
        """
        x = x + self.self_attn(self.input_layernorm(x), positions, kv_cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # One RoPE instance shared across layers: its cos/sin caches depend only on
        # position, and per-layer copies would waste memory
        rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_idx, rotary_emb)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCacheView | None = None,
    ) -> torch.Tensor:
        """Embed the tokens, run every decoder layer, then the final norm.

        Shapes:
            input_ids: [*b, seq_len] int64
            positions: same as input_ids
            ->         [*b, seq_len, hidden_size]
        """
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions, kv_cache)
        return self.norm(x)


class Qwen3ForCausalLM(CausalLM):
    """Qwen3 over a flat [num_tokens] or padded [batch, seq_len] token layout."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load a HF-format state dict into the model."""
        params = self.state_dict()
        for name, tensor in weights.items():
            parts = name.split(".")
            if parts[-2] in SHARD_MAP:
                # HF checkpoints typically store Q, K, V and gate, up projections as separate
                # tensors. We merge these separated shards into single parameters (qkv_proj
                # and gate_up_proj) using the MergedLinear layer, which handles copying each
                # individual tensor into the correct row slice of the fused weight matrix.
                # This enables more efficient, combined matrix multiplications (executing
                # one GEMM instead of multiple) during inference.
                merged_name, index = SHARD_MAP[parts[-2]]
                module = self.get_submodule(".".join([*parts[:-2], merged_name]))
                assert isinstance(module, MergedLinear)
                module.load_shard(index, tensor)
            else:
                params[name].copy_(tensor)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCacheView | None = None,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for every position, or only the `logits_indices` ones.

        Shapes:
            input_ids:      [*b, seq_len] int64
            positions:      same as input_ids
            logits_indices: [num_selected] int64 (token indices, flat layout)
                            | [batch] int64 (one per row, padded layout) | None
            ->              [*b, seq_len, vocab_size]
            -> indexed:     [num_selected, vocab_size] | [batch, vocab_size]
        """
        hidden = self.model(input_ids, positions, kv_cache)
        if logits_indices is not None:
            if hidden.dim() == 2:
                hidden = hidden[logits_indices]
            else:
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                hidden = hidden[rows, logits_indices]
        return self.lm_head(hidden)
