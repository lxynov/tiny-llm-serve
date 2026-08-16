"""A tiny LLM inference and serving engine.

Every method that takes or returns a tensor documents its shapes in a
`Shapes:` block at the end of its docstring -- one line per tensor argument in
signature order, then `->` lines for what the call returns:

    Shapes:
        q:          [*b, q_len, num_heads, head_dim]
        k, v:       [*b, kv_len, num_kv_heads, head_dim]
        attn_mask: ~[*b, 1, q_len, kv_len] bool | None
        ->          same as q
      where kv_len >= q_len

A method whose docstring is only a `Shapes:` block still needs a summary line
above it: `ruff format` re-indents a docstring's body to its least-indented
line, so a block with no prose beside it gets flattened.

Notation:
    [a, b]      exact shape; [] is a scalar tensor
    *b          optional leading batch dim, present only when batched
    ...         any leading dims, passed through unchanged
    ~[...]      only has to broadcast to this, not equal it
    A | B       alternative layouts; `| None` marks an optional tensor
    same        identical to the input above (`-> same as q`)
    bool/int64  dtype, written only when it is not the model's float dtype
    where ...   invariants tying the dims together

Dimension names -- the same name means the same number within one call:
    seq_len                     token positions in this step; per row once
                                batched, so the padded length of a
                                [batch, seq_len] layout
    num_tokens                  a flat layout's whole token axis -- one
                                sequence today, hence the same count as
                                seq_len, but its own name for the day
                                continuous batching packs several into one run
    batch                       rows of a padded step
    q_len, kv_len               attention's query positions and the key/value
                                positions they may attend to
    num_heads, num_kv_heads     attention heads; GQA lets num_kv_heads divide
                                num_heads
    head_dim                    width of one attention head
    hidden_size                 model width; intermediate_size and vocab_size
                                likewise come from ModelConfig
    num_slots, max_model_len    KV pool slots, and the tokens each reserves
    num_seqs                    sequences sampled together
"""

__version__ = "0.1.0"
