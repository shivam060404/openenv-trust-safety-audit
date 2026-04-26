---
title: Trust Safety Audit Env
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: true
license: apache-2.0
---

# COLISEUM Trust & Safety Audit Environment

Multi-agent red-teaming environment with live attacker vs defender simulation.

## Models

| Role | Variant | HF Repo |
|------|---------|---------|
| Defender | SFT Only | [coliseum034/coliseum-defender-sft](https://huggingface.co/coliseum034/coliseum-defender-sft) |
| Defender | SFT + GRPO | [coliseum034/coliseum-defender-grpo-live](https://huggingface.co/coliseum034/coliseum-defender-grpo-live) |
| Attacker | DAN adapter | [coliseum034/coliseum-attacker-dan](https://huggingface.co/coliseum034/coliseum-attacker-dan) |
| Attacker | Wild adapter | [coliseum034/coliseum-attacker-wild](https://huggingface.co/coliseum034/coliseum-attacker-wild) |

Base models: `Qwen/Qwen2.5-1.5B-Instruct` (defender), `Qwen/Qwen2.5-0.5B-Instruct` (attacker)

## Space Secrets

- `HF_TOKEN` — required to download gated models
- `GROQ_API_KEY` — optional, enables victim model responses (llama-3.1-8b-instant)

## Features

- **Manual Audit** — analyse any query with SFT or SFT+GRPO defender
- **Live Arena** — multi-turn attacker vs defender simulation with visual cards
- **SFT vs GRPO Compare** — side-by-side comparison of both defender variants
