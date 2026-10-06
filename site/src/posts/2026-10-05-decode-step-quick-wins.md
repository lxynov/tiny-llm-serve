---
title: Decode step quick wins
date: 2026-10-05
description:
tags: []
draft: false
---

In the [last post](/posts/basic-static-batching/), MBU sat around 0.2 at every batch size.

0.2 is so low that it's kind
of obvious that there's much optimization space even without continuous batching and paged kv cache.

This post notes
down some initial optimizations that I did and their effects.

## Benchmark

Running the benchmark in the last post is costly, so I introduced a new **decode-step microbenchmark** that measures _how
long one decode step takes at a given batch size and KV length_.

Each cell is a `batch x kv_len` pair:

| Cell | Why |
| :--- | :--- |
| `1x576` | Single sequence. 576 is the mean cached length of `uniform-512x128` (512 prompt + half of 128 output) |
| `8x576` | Small batch |
| `64x576` | Medium batch |
| `256x576` | The largest batch that fit in the throughput sweep |
| `64x2048` | Longer context, so the KV cache matters more than the weights |

For each cell, it prefills `batch` random prompts, runs 10 warmup steps, and then times 3 windows of 30 steps. The timed
steps center on `kv_len`. It reports the median window.

Same setup as before: Qwen3-8B in bfloat16 on an H100 80GB.

## Changes

I made four changes, one commit each, and reran the benchmark after each one:

1. [Remove host syncs from KV manager steps](https://github.com/lxynov/tiny-llm-serve/commit/1a431b9).
2. [Remove host syncs from decode steps](https://github.com/lxynov/tiny-llm-serve/commit/67c4bdd).
3. [Round decode's `kv_len` up to a bucket](https://github.com/lxynov/tiny-llm-serve/commit/dc1e058).
4. [Read decode's KV as a view when its slots are contiguous](https://github.com/lxynov/tiny-llm-serve/commit/26d95f6).

## Results

Milliseconds per decode step:

| Cell | Baseline | 1. KV manager syncs | 2. Decode syncs | 3. `kv_len` bucket | 4. KV view | Total speedup |
|---|---|---|---|---|---|---|
| `1x576` | 56.9 | 58.9 | 61.9 | 23.9 | 22.8 | 2.5x |
| `8x576` | 57.7 | 61.8 | 62.1 | 23.6 | 22.8 | 2.5x |
| `64x576` | 58.7 | 61.9 | 62.8 | 24.0 | 22.8 | 2.6x |
| `256x576` | 76.2 | 76.8 | 74.0 | 41.7 | 25.3 | 3.0x |
| `64x2048` | 69.4 | 69.0 | 69.2 | 35.0 | 22.7 | 3.1x |

MBU:

| Cell | Baseline | 1. KV manager syncs | 2. Decode syncs | 3. `kv_len` bucket | 4. KV view |
|---|---|---|---|---|---|
| `1x576` | 0.086 | 0.083 | 0.079 | 0.206 | 0.215 |
| `8x576` | 0.088 | 0.083 | 0.082 | 0.215 | 0.224 |
| `64x576` | 0.110 | 0.107 | 0.104 | 0.271 | 0.290 |
| `256x576` | 0.149 | 0.148 | 0.154 | 0.265 | 0.437 |
| `64x2048` | 0.155 | 0.156 | 0.155 | 0.307 | 0.474 |


## Surprising Findings

1. **Removing host syncs didn't make anything faster.** At small batches it even looked 3–8% slower, but with one run
   per commit I can't tell that apart from noise. In hindsight this makes sense. A host sync is expensive when the GPU
   has a long queue of work that the CPU could have kept adding to. Here the GPU wasn't the bottleneck: the step took
   ~57 ms at batch 1 through 64, while reading the weights once takes about 5 ms. The GPU was mostly waiting on the host
   anyway, so waiting for it to drain cost little. These commits are still needed, because CUDA graphs can't capture a
   step that syncs. The payoff should come later.

2. **Bucketing `kv_len` cut about 34 ms from every cell**, at batch 1 just as much as at batch 256. A saving that doesn't
   depend on batch size isn't about bytes. It's a fixed cost per step, and it went away once steps stopped having new
   shapes. There's a second piece of evidence. In `64x2048`, the middle window runs from kv_len 2034 to 2063, so it
   crosses from the 2048 bucket into the 2176 bucket. After bucketing, that window is consistently ~1.3 ms/step slower
   than its neighbors, which is ~40 ms over 30 steps, paid once. So a never-seen-before attention shape costs roughly
   35–40 ms, and before bucketing every step paid it.

   <!-- TODO: what pays the per-shape cost? Profile one step at 67c4bdd vs dc1e058. -->
   I haven't pinned down what pays that cost yet. That's the next thing to profile.

3. **Reading the KV as a view saved exactly the copy it removed.** One token's K and V across all 36 layers is
   147,456 bytes. At `256x576`, with `kv_len` bucketed to 640, the old gather copied
   256 × 640 × 147,456 B ≈ 24 GB per step, which is 48 GB read and written. At the H100's 3.35 TB/s, that's ~14 ms, and
   the step got 16.4 ms faster. At `64x2048` the same estimate gives ~11.5 ms, and the step got 12.4 ms faster. At small
   batches the copy is small, and the gain was about 1 ms.

4. **The step is now host-bound.** Every cell from `1x576` to `64x2048` takes ~22.7 ms. Reading the weights once at
   batch 1 should take ~4.9 ms (16.4 GB at 3.35 TB/s). When the time doesn't move as the batch grows 64x, the GPU isn't
   what we're waiting on. The remaining ~18 ms is most likely Python plus kernel launches: in eager mode, each of the 36
   layers launches its kernels one by one. That's what CUDA graphs are for, and the last three changes are what make
   them possible.
