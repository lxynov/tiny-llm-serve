---
title: Basic static batching
date: 2026-08-24
description:
tags: []
draft: false
---

I've been working on this side project for a while. As I'm taking one week of PTO from work to recharge and experience
US Open Fan Week, I'd also like to take some time to reflect and plan for this project.

A quick recap: this project is to build an educational LLM serving engine in a benchmark-driven way.

## Implementation

What's achieved so far is an inference engine supporting basic static batch processing. (Browse the code: [4f98bb1](https://github.com/lxynov/tiny-llm-serve/tree/4f98bb172bc10dea5a3fc009ddea1febc178e1be))

The primitives upon which it was built are:
- PyTorch. It uses the common tensor and compute library. One key usage is PyTorch's [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html)
  function for attention computation.
- Hugging Face libraries including `transformers`, `safetensors`, and `huggingface-hub`. The `transformers` libray is
  used for tokenization, model loading, and correctness testing, instead of directly for inference.

It currently supports Qwen3 dense models only.

It runs everything on a single GPU.

It's a library so far instead of a service. An imaginary user would call [LLM.generate_batch_ids()](https://github.com/lxynov/tiny-llm-serve/blob/4f98bb172bc10dea5a3fc009ddea1febc178e1be/src/tiny_llm_serve/engine/llm_engine.py#L107)
to generate outputs for a batch of prompts.

It's **static batching** in the sense that a batch of prompts is processed in an "atomic" step. A finished request stays
in the batch until every request has completed.

KV cache is simple in static batching. You just preallocate a buffer for the entire batch.

## Benchmark

I benchmarked the implementation with an H100 80GB GPU, rented via [runpod.io](https://runpod.io/).

Because the existing implementation is a library rather than a service, I mostly benchmarked the throughput instead of
latency, using offline workloads. Once the service is implemented, I will add trace-drive workloads where each request
is timestamped, and more metrics around latency.

Here are the three workloads:
| Workload | Prompt Length | Output Length | Measurement |
| :--- | :--- | :--- | :--- |
| `uniform-512x128` | Fixed 512 | Fixed 128 | How the engine performs when there's no waste | 
| `mixed-out` | Fixed 512 | Log-normal | Waste due to output variance — completed requests wait for others during decoding |
| `sharegpt-like` | Log-normal | Log-normal | Waste due to input variance — padding during prefill |

Each workload has 512 requests and is tested with 10 batch sizes.

The model used is Qwen3-8B in bfloat16.

## Results

| Column      | Meaning                                                                                                                                                 |
|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bs`        | Batch size                                                                                                                                              |
| `status`    | `ok`, `oom`, or `not attempted`                                                                                                                         |
| `wall_s`    | Wall time                                                                                                                                               |
| `pre%`      | Share of wall time spent in prefill                                                                                                                     |
| `ms/step`   | Milliseconds per decode step, averaged over the pass                                                                                                    |
| `out_tok/s` | Generated tokens ÷ `wall_s`, so prefill time is in the denominator                                                                                       |
| `speedup`   | `out_tok/s` against the `bs = 1`                                                                                                                        |
| `occ`       | Decode occupancy. (`output_tokens` - 512) ÷ (`bs` × `decode_steps`), where `-512` is because each request's first token comes out of the prefill stage. |
| `kv_eff`    | KV cache efficiency. KV tokens used ÷ KV tokens reserved. A batch reserves `bs` × (`longest prompt` + `longest output`) token slots.                    |
| `mbu`       | Memory bandwidth utilization in decode. the bytes decode must read (all the weights once per step, plus each sequence's KV window) ÷ decode time ÷ the H100's peak HBM bandwidth                                                                                                                |
| `peak_GB`   | Peak GPU memory usage                                                                                                                                   |

### uniform-512x128

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 1539.5 | 0.9% | 23.5 | 42.6 | 1.0 | 1.000 | 1.000 | 0.209 | 15.5 |
| 2 | ok | 758.5 | 1.4% | 23.0 | 86.4 | 2.0 | 1.000 | 1.000 | 0.215 | 15.7 |
| 4 | ok | 393.4 | 2.4% | 23.6 | 166.6 | 3.9 | 1.000 | 1.000 | 0.211 | 16.0 |
| 8 | ok | 208.1 | 4.3% | 24.5 | 315.0 | 7.4 | 1.000 | 1.000 | 0.208 | 16.6 |
| 16 | ok | 122.4 | 7.2% | 28.0 | 535.5 | 12.6 | 1.000 | 1.000 | 0.189 | 18.0 |
| 32 | ok | 63.8 | 13.7% | 27.1 | 1026.6 | 24.1 | 1.000 | 1.000 | 0.210 | 20.7 |
| 64 | ok | 38.2 | 22.9% | 29.0 | 1714.3 | 40.3 | 1.000 | 1.000 | 0.225 | 26.2 |
| 128 | ok | 28.4 | 32.1% | 38.0 | 2305.0 | 54.1 | 1.000 | 1.000 | 0.214 | 37.0 |
| 256 | ok | 24.8 | 35.9% | 62.6 | 2638.5 | 62.0 | 1.000 | 1.000 | 0.182 | 58.8 |
| 512 | oom | - | - | - | - | - | - | - | - | - |

### mixed-out

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 1229.1 | 1.0% | 23.8 | 42.0 | 1.0 | 1.000 | 1.000 | 0.207 | 15.5 |
| 2 | ok | 922.1 | 1.2% | 23.7 | 56.0 | 1.3 | 0.664 | 0.924 | 0.209 | 15.8 |
| 4 | ok | 649.8 | 1.5% | 23.9 | 79.4 | 1.9 | 0.477 | 0.849 | 0.209 | 16.2 |
| 8 | ok | 448.1 | 2.0% | 24.5 | 115.2 | 2.7 | 0.357 | 0.773 | 0.209 | 17.3 |
| 16 | ok | 315.8 | 2.8% | 25.5 | 163.5 | 3.9 | 0.265 | 0.689 | 0.212 | 21.0 |
| 32 | ok | 199.9 | 4.4% | 26.3 | 258.3 | 6.1 | 0.220 | 0.633 | 0.226 | 26.7 |
| 64 | ok | 123.9 | 7.1% | 29.5 | 416.8 | 9.9 | 0.205 | 0.612 | 0.238 | 29.5 |
| 128 | ok | 91.7 | 9.6% | 40.6 | 562.9 | 13.4 | 0.195 | 0.598 | 0.227 | 43.8 |
| 256 | ok | 78.5 | 11.2% | 68.2 | 657.9 | 15.7 | 0.195 | 0.598 | 0.199 | 72.3 |
| 512 | oom | - | - | - | - | - | - | - | - | - |

### sharegpt-like

| bs | status | wall_s | pre% | ms/step | out_tok/s | speedup | occ | kv_eff | mbu | peak_GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 2372.7 | 1.0% | 24.1 | 41.2 | 1.0 | 1.000 | 1.000 | 0.203 | 15.8 |
| 2 | ok | 1778.1 | 0.9% | 24.4 | 55.0 | 1.3 | 0.675 | 0.673 | 0.202 | 16.4 |
| 4 | ok | 1329.5 | 0.9% | 25.2 | 73.5 | 1.8 | 0.465 | 0.471 | 0.199 | 17.4 |
| 8 | ok | 892.6 | 1.3% | 25.6 | 109.5 | 2.7 | 0.354 | 0.355 | 0.203 | 22.5 |
| 16 | ok | 623.1 | 2.0% | 26.6 | 156.9 | 3.8 | 0.265 | 0.269 | 0.213 | 25.4 |
| 32 | ok | 411.3 | 3.5% | 28.9 | 237.7 | 5.8 | 0.221 | 0.222 | 0.232 | 29.5 |
| 64 | ok | 284.8 | 5.6% | 36.3 | 343.3 | 8.3 | 0.205 | 0.201 | 0.245 | 43.8 |
| 128 | ok | 247.8 | 6.8% | 56.4 | 394.5 | 9.6 | 0.186 | 0.185 | 0.238 | 72.3 |
| 256 | oom | - | - | - | - | - | - | - | - | - |
| 512 | not attempted | - | - | - | - | - | - | - | - | - |

## Findings

1. Increasing the batch size improves throughput — initially linearly, then with diminishing returns, and ultimately leading to OOMs.

![Output tokens per second against batch size, on log-log axes.](/images/qwen3-8b-bfloat16-n512-seed0-cuda-throughput-{theme}.png)

2. MBU (Memory Bandwidth Utilization) is consistently low across all the batch sizes and workloads, ranging from 0.18
   to 0.25. Note that decoding is memory-bound, so the memory access latency directly impacts performance. However, only
   20% of memory bandwidth is utilized.

3. PyTorch's `scaled_dot_product_attention` function has several backend implementations, and it turned out that the
   current code's decoding path actually uses the math one, which is not the efficient one. The FlashAttention backend doesn't accept 
   `attn_mask` that the current code passes. Note that people have valid reasons to disable FlashAttention when an
   `attn_mask` is present, because FlashAttention's goal is to avoid the materialization of full score matrix, but the
   `attn_mask` itself has the same shape of the score matrix. To confirm that `scaled_dot_product_attention` is indeed
   using the math one, I profiled one decode step with PyTorch profiler, and indeed we could see:

   ```
   Name                                       Self CUDA   CUDA total   # of Calls   Input Shapes
   aten::scaled_dot_product_attention            0.000us    24.810ms           28   [[64, 16, 1, 128], [64, 8, 513, 128], ...
   aten::_scaled_dot_product_attention_math      0.000us    24.679ms           28   [[64, 16, 1, 128], [64, 8, 513, 128], ...
   aten::repeat_interleave                       0.000us    11.954ms           56   [[64, 8, 513, 128], [], [], []]
   aten::copy_                                  11.954ms    11.954ms           56   [[64, 8, 2, 513, 128], ...
   aten::mul                                     5.012ms     5.012ms           28   [[64, 16, 128, 513], []]
   aten::bmm                                     4.059ms     4.351ms           28   [[1024, 1, 128], [1024, 128, 513]]
   aten::bmm                                     2.735ms     2.834ms           28   [[1024, 1, 513], [1024, 513, 128]]

   Self CUDA time total: 35.238ms
   ```
