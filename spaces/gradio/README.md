---
title: Nuvo-300M
emoji: 🐣
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Nuvo-300M

A ~299M parameter language model, trained from scratch on 6B tokens and
instruction-tuned on Stanford Alpaca.

**Architecture:** GQA + RoPE + RMSNorm + SwiGLU, decoder-only transformer.

**Training:** Modal H100, AdamW + cosine schedule, then Alpaca SFT.

**Source:** [github.com/tajbaba999/nuvo_v1](https://github.com/tajbaba999/nuvo_v1)
