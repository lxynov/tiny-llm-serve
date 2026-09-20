#!/usr/bin/env bash
# Runs static/batched decode with the profiler hook in llm_engine.py enabled
# (TINY_LLM_SERVE_DEBUG_PROFILE_DECODE=1), so the kernel table for one decode
# step prints to stderr. This is the check behind each step of
# docs/before-continuous-batching.md: which SDPA backend decode dispatches to,
# and whether the copies §1.2 measured are gone.
#
# Defaults to bfloat16, which is what the plan targets. Pass float32 to
# reproduce the original fp32 profile in §1.2.
set -euo pipefail

TINY_LLM_SERVE_DEBUG_PROFILE_DECODE=1 python -m benchmarks.bench_throughput \
  --model Qwen/Qwen3-0.6B --workload uniform-512x128 \
  --mode static --batch-size 64 --num-requests 64 --dtype "${1:-bfloat16}"
