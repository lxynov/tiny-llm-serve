#!/usr/bin/env bash
# Runs static/batched decode with the profiler hook in llm_engine.py enabled
# (TINY_LLM_SERVE_DEBUG_PROFILE_DECODE=1), so the kernel table for one decode
# step prints to stderr. See docs/raising-decode-mbu.md item 1: which SDPA
# backend decode actually dispatches to.
set -euo pipefail

TINY_LLM_SERVE_DEBUG_PROFILE_DECODE=1 python -m benchmarks.bench_throughput \
  --model Qwen/Qwen3-0.6B --workload uniform-512x128 \
  --mode static --batch-size 64 --num-requests 64
