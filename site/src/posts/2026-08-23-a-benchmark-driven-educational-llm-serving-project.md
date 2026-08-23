---
title: A benchmark-driven educational LLM serving project
date: 2026-08-23
description:
tags: []
draft: false
---

I've been spending time learning about LLM systems. My efforts include:
- Took the [Large Language Model Systems](https://llmsystem.github.io/) course as part of
[the CMU GenAI&LLM certificate program](https://www.cmu.edu/online/generative-ai-llms).
- Vibe wrote [an AI system book](https://lxynov.github.io/aisys-book/), which covers topics ranging from hardware
to the software layers for training/inference.
- Watched lectures from Stanford [CS336](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV).
- Read classical papers (Orca, PagedAttention, FlashAttention, SGLang, etc.).

They were interesting pastimes, but a more fun thing to do is to build an inference engine from scratch, so I started
this project.

I plan to cover some of the most important concepts in LLM serving – e.g., continuous batching, paged KV cache, an
OpenAI-compatible API, etc.

Like other systems, performance is a key part of an LLM serving system. So I decided to structure this project in a
**benchmark-driven** way: I won't just implement everything all at once as if I have the foresight. Instead, I'll start
with the most basic implementation, measure its performance, and then apply the optimizations to gradually improve it.

