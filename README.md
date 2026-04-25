---
title: OpenEnv Trust Safety Arena
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# OpenEnv Trust & Safety Arena (HF Space)

Interactive demo for a production-grade RL trust-and-safety system:
- Multi-agent attacker vs defender arena
- GRPO-trained defender policy
- Anti-reward-hacking reward components
- Before vs after training evidence for judges

## What the demo shows

1. **Live Arena**
   - Run one episode with user prompt or auto-generated attack
   - See attacker prompt, defender decision (`ALLOW`/`BLOCK`), reasoning, reward breakdown
   - Color-coded outcomes: Green=TP, Red=FN, Yellow=FP

2. **Attack Simulator**
   - Choose attack family: `DAN`, `Wild`, `Mutation`
   - Choose difficulty tier
   - Watch multi-turn evolution and mutation trail

3. **Training Insights**
   - Before/after curves (baseline -> post-SFT -> post-GRPO)
   - Confusion matrix
   - Attack suppression trend (1 - false negative rate)

4. **Before vs After**
   - Run same attack against baseline and trained defender
   - Side-by-side decision + rationale comparison

5. **System Explanation**
   - Non-technical summary for judges
   - 2-minute demo flow script for video/live pitch

## Architecture

- `app.py`: Gradio UI and interaction logic
- `environment.py`: OpenEnv-compatible arena with `reset`, `step`, reward shaping
- `run_arena.py`: Local terminal arena runner for full interaction logs
- `server.py`: FastAPI/OpenEnv API (`/reset`, `/step`, `/state`, compatibility routes)
- `coliseum_defender/eval/run_evaluation.py`: evaluation + confusion matrix generation

### Backend integration used in the app

- `run_single_episode(...)`: wrapper around one arena decision cycle
- `simulate_attack(...)`: multi-turn attacker evolution simulator
- `get_reward_breakdown(...)`: reward component extraction from env step payload
- OpenEnv API preference: `OPENENV_URL` (`http://localhost:7860` default)
- Defender API preference: `DEFENDER_API_URL` (`http://localhost:8001` default)
- Automatic local fallback if APIs are unavailable (CPU-friendly for Spaces)

## Hugging Face Spaces deployment

1. Push repository to a new Space (SDK = Gradio).
2. Ensure `README.md` YAML block is present (already included above).
3. Dependencies are installed from `requirements.txt`.
4. Optional Space secrets:
   - `HF_TOKEN`
   - `OPENAI_API_KEY` or `GROQ_API_KEY`
   - `OPENENV_URL` (if using an external env service)
   - `DEFENDER_API_URL` (if using external trained defender API)
5. Space starts `app.py` automatically.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Demo storytelling script (2 minutes)

1. **Problem setup**: "Here is a malicious prompt; baseline defender misses or hesitates."
2. **RL impact**: "Now the GRPO-trained defender blocks the same attack."
3. **Evidence**: "F1 and confusion matrix improve after training."
4. **Robustness**: "Reward is multi-component, so reward hacking is penalized and safety behavior stays stable."

## Performance notes

- Designed to run without GPU
- Uses lightweight local logic when model APIs are unavailable
- Pulls precomputed eval metrics if present, else uses safe fallback demo metrics
