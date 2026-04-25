"""
download_models.py
==================
Downloads all COLISEUM models to models/ so inference is instant (no runtime HF calls).

Run once before anything else:
    source env/bin/activate
    python download_models.py

After this, all scripts load from local disk automatically.

Directory layout created:
    models/
      base/
        Qwen2.5-0.5B-Instruct/      ← attacker base model
        Qwen2.5-1.5B-Instruct/      ← defender base model
      adapters/
        coliseum-attacker-dan/       ← DAN LoRA adapter
        coliseum-attacker-wild/      ← WildTeam LoRA adapter
        coliseum-defender-sft/       ← Defender SFT LoRA adapter
"""

import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("[ERROR] HF_TOKEN not set. Add it to your .env file.")
    sys.exit(1)

from huggingface_hub import snapshot_download

MODELS_DIR = Path(__file__).parent / "models"
BASE_DIR   = MODELS_DIR / "base"
ADPT_DIR   = MODELS_DIR / "adapters"
BASE_DIR.mkdir(parents=True, exist_ok=True)
ADPT_DIR.mkdir(parents=True, exist_ok=True)

# ── What to download ──────────────────────────────────────────────────────────
DOWNLOADS = [
    # (hf_repo_id,                          local_dir,                                    label)
    ("Qwen/Qwen2.5-0.5B-Instruct",          BASE_DIR / "Qwen2.5-0.5B-Instruct",          "Attacker base model (0.5B)"),
    ("Qwen/Qwen2.5-1.5B-Instruct",          BASE_DIR / "Qwen2.5-1.5B-Instruct",          "Defender base model (1.5B)"),
    ("coliseum034/coliseum-attacker-dan",    ADPT_DIR / "coliseum-attacker-dan",           "DAN attacker LoRA adapter"),
    ("coliseum034/coliseum-attacker-wild",   ADPT_DIR / "coliseum-attacker-wild",          "WildTeam attacker LoRA adapter"),
    ("okaditya08/coliseum-defender-sft",     ADPT_DIR / "coliseum-defender-sft",           "Defender SFT LoRA adapter"),
]

def download(repo_id: str, local_dir: Path, label: str) -> None:
    local_dir = Path(local_dir)
    # Already downloaded if directory has files
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"  [SKIP] {label} — already at {local_dir}")
        return

    print(f"\n  Downloading: {label}")
    print(f"    HF repo  : {repo_id}")
    print(f"    Local dir: {local_dir}")
    t0 = time.time()
    snapshot_download(
        repo_id   = repo_id,
        local_dir = str(local_dir),
        token     = HF_TOKEN,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file()) / 1e6
    print(f"    Done in {elapsed:.0f}s — {size_mb:.0f} MB on disk")


if __name__ == "__main__":
    print("=" * 60)
    print("  COLISEUM — Download All Models")
    print("=" * 60)
    print(f"  Saving to: {MODELS_DIR.resolve()}")
    print(f"  HF Token : {'SET ✅' if HF_TOKEN else 'NOT SET ❌'}")
    print()

    for repo_id, local_dir, label in DOWNLOADS:
        download(repo_id, local_dir, label)

    print("\n" + "=" * 60)
    print("  All models downloaded.")
    print(f"  Total size: {sum(f.stat().st_size for f in MODELS_DIR.rglob('*') if f.is_file()) / 1e6:.0f} MB")
    print()
    print("  Now run:")
    print("    python run_arena.py --skip_attackers --skip_defender  # test flow")
    print("    python run_arena.py --skip_attackers                  # real defender")
    print("    python run_arena.py                                   # everything real")
    print("=" * 60)
