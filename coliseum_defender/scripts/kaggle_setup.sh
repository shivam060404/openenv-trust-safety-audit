#!/usr/bin/env bash
# scripts/kaggle_setup.sh
# ========================
# Run this in a Kaggle notebook shell cell to set up the project.
# Usage (in Kaggle): !bash scripts/kaggle_setup.sh
#
# What it does:
#   1. Clones the GitHub repo
#   2. Installs all Python dependencies
#   3. Validates GPU and environment
#   4. Prints the run order for notebooks

set -euo pipefail

REPO_URL="https://github.com/shivam060404/openenv-trust-safety-audit"  # UPDATE THIS
WORK_DIR="/kaggle/working"
PROJECT_DIR="${WORK_DIR}/coliseum"

echo "══════════════════════════════════════════════"
echo "  COLISEUM Defender — Kaggle Setup"
echo "══════════════════════════════════════════════"

# ── 1. GPU check ──────────────────────────────────
echo ""
echo "── GPU Status ──"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name}')
    print(f'  VRAM: {vram:.1f} GB')
    if vram < 14:
        print('  ⚠️  WARNING: Less than 14GB VRAM. Use T4 x1 accelerator.')
    else:
        print('  ✅ VRAM OK')
else:
    print('  ❌ No GPU! Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x1')
    exit(1)
"

# ── 2. Clone repo ─────────────────────────────────
echo ""
echo "── Cloning Repository ──"
if [ -d "${PROJECT_DIR}" ]; then
    echo "  Already exists — pulling latest..."
    cd "${PROJECT_DIR}" && git pull
else
    git clone "${REPO_URL}" "${PROJECT_DIR}"
    echo "  ✅ Cloned to ${PROJECT_DIR}"
fi
cd "${PROJECT_DIR}"

# ── 3. Install dependencies ───────────────────────
echo ""
echo "── Installing Python Packages ──"
echo "  (This takes 3-5 minutes)"

# Core ML deps
pip install -q \
    "datasets>=2.18.0" \
    "transformers>=4.43.0" \
    "accelerate>=0.28.0" \
    "bitsandbytes>=0.42.0" \
    "peft>=0.10.0" \
    "trl==0.15.2" \
    "huggingface_hub>=0.22.0"

# Unsloth (Kaggle-specific install)
pip install -q unsloth 2>/dev/null || \
    pip install -q "unsloth[colab-new]" 2>/dev/null || \
    echo "  ⚠️  Unsloth install issues — will use transformers fallback"

# Utility
pip install -q \
    wandb \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    pandas \
    fastapi \
    uvicorn \
    requests

echo "  ✅ Packages installed"

# ── 4. Check HF token ─────────────────────────────
echo ""
echo "── Checking HuggingFace Token ──"
python3 -c "
import os
token = os.environ.get('HF_TOKEN', '')
if token:
    print(f'  ✅ HF_TOKEN is set ({len(token)} chars)')
else:
    print('  ⚠️  HF_TOKEN not set as Kaggle secret.')
    print('     Add it: Settings → Add-ons → Secrets → Add HF_TOKEN')
    print('     Required for: LlamaGuard download, HF Hub push')
"

# ── 5. Verify project structure ───────────────────
echo ""
echo "── Project Files ──"
find "${PROJECT_DIR}" -name "*.py" -o -name "*.ipynb" | sort | while read f; do
    echo "  ${f#${PROJECT_DIR}/}"
done

# ── 6. Print run order ────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo "  NOTEBOOK RUN ORDER"
echo "══════════════════════════════════════════════"
echo ""
echo "  📅 PRE-HACKATHON (Wed Apr 23 — Today)"
echo "  ┌─ 01_data_preprocessing.ipynb"
echo "  │   → Downloads 3 datasets + runs LlamaGuard teacher"
echo "  │   → Output: defender_train.jsonl, defender_eval.jsonl"
echo "  │   → Time: ~3 hours (teacher labeling ~5K samples)"
echo "  │"
echo "  └─ 02_defender_sft.ipynb  (Thu Apr 24)"
echo "      → SFT with distillation labels"
echo "      → Output: coliseum-defender-sft on HF Hub"
echo "      → Time: ~1 hour (3 epochs, T4)"
echo ""
echo "  🚀 HACKATHON DAY (Sat Apr 26)"
echo "  ┌─ 03_grpo_training.ipynb"
echo "  │   → GRPO in-environment RL (500 steps)"
echo "  │   → Output: coliseum-defender-grpo + reward curve PNG"
echo "  │   → Time: ~90 minutes"
echo "  │"
echo "  └─ Run defender API + eval"
echo "      python integration/defender_api.py --model_repo <grpo_repo>"
echo "      python eval/run_evaluation.py --stage post_grpo ..."
echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ Setup complete. Open notebook 01 to start."
echo "══════════════════════════════════════════════"
