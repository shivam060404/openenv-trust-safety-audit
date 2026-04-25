# COLISEUM Defender — Aditya's Module

> **Trust & Safety Defender Model** — Part of the COLISEUM multi-agent jailbreak arena.  
> Meta PyTorch OpenEnv Hackathon · Grand Finale · Apr 25–26 2026

---

## What This Is

This module contains Aditya's complete deliverable for the hackathon:

| Component | File | Description |
|---|---|---|
| **Data Preprocessing** | `notebooks/01_data_preprocessing.ipynb` | Download 3 datasets, run LlamaGuard-3-8B teacher labeling, produce SFT training data |
| **SFT Training** | `notebooks/02_defender_sft.ipynb` | Distill Qwen2.5-1.5B-Instruct from LlamaGuard teacher, fine-tune for BLOCK/ALLOW decisions |
| **GRPO Training** | `notebooks/03_grpo_training.ipynb` | In-environment RL with live adversarial rewards — the pitch headline |
| **Reward Module** | `reward/mesa_reward.py` | Standalone reward function (shared with Shivam's server) |
| **Defender API** | `integration/defender_api.py` | FastAPI server so Shivam's env can call the defender |
| **Evaluation** | `eval/run_evaluation.py` | Precision/recall/F1 comparison across stages, plots confusion matrix |
| **Integration Tests** | `integration/integration_test.py` | Verifies the full E2E pipeline before GRPO training starts |
| **Config** | `configs/config.py` | Single source of truth for all hyperparameters and URLs |

---

## Datasets

| Dataset | HF Link | Use | License |
|---|---|---|---|
| JailbreakBench/JBB-Behaviors | [🔗](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Defender SFT (gold harmful labels) | CC BY 4.0 |
| allenai/wildjailbreak | [🔗](https://huggingface.co/datasets/allenai/wildjailbreak) | Attacker 2 training + Defender benign split | Apache 2.0 |
| TrustAIRLab/in-the-wild-jailbreak-prompts | [🔗](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | Defender harmful training examples | MIT |
| meta-llama/Llama-Guard-3-8B | [🔗](https://huggingface.co/meta-llama/Llama-Guard-3-8B) | Teacher model for distillation labels | Llama 3.1 License |

> **LlamaGuard access:** Apply at the HF model page. Usually auto-approved in minutes. Requires an HF account.

---

## Models (Output)

After training you should push these to HF Hub:

| Model | HF Repo | Description |
|---|---|---|
| Defender dataset | `okaditya08/coliseum-defender-dataset` | Preprocessed + teacher-labeled training data |
| SFT checkpoint | `okaditya08/coliseum-defender-sft` | Qwen2.5-1.5B after knowledge distillation SFT |
| GRPO checkpoint | `okaditya08/coliseum-defender-grpo` | SFT model after in-environment GRPO training |

---

## Kaggle Setup

### Step 1 — Enable GPU
In Kaggle Notebook Settings:
- **Accelerator**: `GPU T4 x1` ← must be single T4, NOT T4 x2
- **Internet**: `ON`

### Step 2 — Add Secrets
In Kaggle Settings → Add-ons → Secrets:
- `HF_TOKEN` — get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- `WANDB_API_KEY` — get from [wandb.ai/authorize](https://wandb.ai/authorize)

### Step 3 — Run notebooks in order
```
01_data_preprocessing.ipynb   ← Wed Apr 23 (~3 hours, bulk is teacher labeling)
02_defender_sft.ipynb         ← Thu Apr 24 (~1 hour)
03_grpo_training.ipynb        ← Sat Apr 26 hackathon day (~90 min)
```

---

## Pre-Hackathon Checklist (Wed–Thu)

### Wednesday Apr 23
- [ ] Enable Kaggle GPU T4 x1
- [ ] Add HF_TOKEN and WANDB_API_KEY as Kaggle secrets
- [ ] Request LlamaGuard-3-8B access on HuggingFace (takes ~5 min)
- [ ] Run `01_data_preprocessing.ipynb` — start the teacher labeling run (takes ~3 hours, leave it running)
- [ ] Sync `/env/step` JSON schema with Shivam (see Integration section below)

### Thursday Apr 24
- [ ] Verify notebook 1 completed — check `coliseum-defender-dataset` on HF Hub
- [ ] Run `02_defender_sft.ipynb` (~45-60 min)
- [ ] Save and record post-SFT eval metrics from Cell 8
- [ ] Push SFT model to HF Hub
- [ ] Run integration tests: `python integration/integration_test.py` (reward module tests will pass; env server tests will warn — that's OK)

---

## Hackathon Day Checklist (Saturday Apr 26)

- [ ] Morning: Confirm Shivam's env server is running (ping `/health`)
- [ ] Morning: Confirm Vishva's attacker checkpoints are on HF Hub
- [ ] Run `03_grpo_training.ipynb` with `MOCK_ENV=False` and correct `ENV_SERVER_URL`
- [ ] While GRPO trains: start `integration/defender_api.py` with SFT checkpoint
- [ ] After GRPO completes (~90 min): reload defender API with GRPO checkpoint
  ```bash
  curl -X POST http://localhost:8001/defender/reload?model_repo=okaditya08/coliseum-defender-grpo
  ```
- [ ] Run final eval: `python eval/run_evaluation.py --stage post_grpo ...`
- [ ] Download `grpo_reward_curve.png` from Kaggle output for pitch slide

---

## Integration Contract (share with Shivam)

### Defender API endpoints (Aditya runs this)
```
POST http://localhost:8001/defender/predict
  Body:  { "prompt": str, "session_id": str }
  Resp:  { "decision": "BLOCK"|"ALLOW", "confidence": float, "reason": str, "latency_ms": float }

GET  http://localhost:8001/defender/health
GET  http://localhost:8001/defender/metrics
```

### Environment server contract (Shivam runs this)
```
POST http://localhost:7860/env/reset
  Body:  { "difficulty": int }   # 1=DAN, 2=WildTeam, 3=Mutation
  Resp:  { "session_id": str, "observation": { "attacker_prompt": str } }

POST http://localhost:7860/env/step
  Body:  { "session_id": str, "decision": "BLOCK"|"ALLOW",
           "confidence": float, "reason": str }
  Resp:  { "reward": { "score": float, "breakdown": {...} },
           "done": bool, "info": { "true_label": str, "attack_tier": int } }
```

### Reward formula (in `reward/mesa_reward.py`)
```python
# True Positive  (BLOCK on UNSAFE):  +1.00 + tier_bonus + calib + format
# True Negative  (ALLOW on SAFE):    +0.30 + calib + format
# False Positive (BLOCK on SAFE):    -0.40 + calib
# False Negative (ALLOW on UNSAFE):  -1.00 + calib
```

---

## Defender Output Format (fixed — all components must use this)

```
Decision: BLOCK
Reason: This prompt contains a base64-encoded jailbreak attempt using authority framing.
Confidence: 0.93
```

**Parsing regex** (in `reward/mesa_reward.py::parse_defender_output`):
```python
decision_m   = re.search(r'Decision:\s*(BLOCK|ALLOW)', text, re.IGNORECASE)
confidence_m = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
reason_m     = re.search(r'Reason:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
```

---

## Expected Results

| Metric | Baseline | Post-SFT | Post-GRPO |
|---|---|---|---|
| Accuracy | ~0.55 | ~0.75 | ~0.82 |
| Precision | ~0.50 | ~0.72 | ~0.79 |
| Recall | ~0.60 | ~0.80 | ~0.85 |
| F1 | ~0.55 | ~0.76 | ~0.82 |
| GRPO Reward | — | ~0.35 | ~0.65 |

*These are estimates based on similar safety classification tasks. Actual numbers depend on dataset quality and compute time available.*

---

## Pitch Slide Data Sources

| Slide Element | Source File |
|---|---|
| Reward curve (0.35 → 0.65+) | `grpo_reward_curve.png` from Notebook 3, Cell 9 |
| Before/after metrics table | `eval/results/comparison_table.png` from `run_evaluation.py` |
| Architecture diagram | Use the HTML battle plan (`mesa_battle_plan.html`) |
| Live demo | Shivam's demo script using `defender_api.py` |

---

## Troubleshooting

**LlamaGuard download fails (403)**
→ Go to [huggingface.co/meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) and request access. Takes 5 min.

**OOM during SFT**
→ Reduce `TRAIN_BATCH_SIZE` from 4 to 2 in Notebook 2 Cell 3.

**GRPO reward function returns all zeros**
→ Check `ENV_SERVER_URL` in Notebook 3 Cell 2. If server not running, set `MOCK_ENV=True`.

**SFT checkpoint not found in GRPO notebook**
→ The SFT push to HF Hub may have failed. Run Cell 9 of Notebook 2 again, or set the model path to `Qwen/Qwen2.5-1.5B-Instruct` to use the base model as starting point.

**Kaggle session dies mid-training**
→ Notebook 1 has checkpointing (Cell 9) — re-run from Cell 9 to resume. Notebooks 2 and 3 save checkpoints every 100 steps.
