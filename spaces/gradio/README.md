---
title: Nuvo-30M
emoji: 🐣
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Nuvo-30M

A 30M parameter language model, trained from scratch on TinyStories.

**Architecture:** GQA + RoPE + RMSNorm + SwiGLU, decoder-only transformer.

**Training:** Modal A10G, 60k steps, AdamW + cosine schedule.

**Source:** [github.com/tajbaba999/nuvo_v1](https://github.com/tajbaba999/nuvo_v1)
