"""
GRPO training entry point for the OpenEnv trust-safety arena.

Modes:
  collect  – run the live environment and write GRPO-ready JSONL trajectories.
             Uses local attacker models (Qwen2.5-0.5B + DAN/Wild adapters) and
             a local deterministic victim — no API calls required.
  train    – run TRL GRPOTrainer on the local Qwen2.5-1.5B + SFT adapter.
             Strategy: merge SFT adapter into base weights → load merged model
             → add a fresh trainable LoRA via PEFT → GRPOTrainer.
             This avoids any Unsloth tensor-wrapper incompatibilities and works
             fully on-device (CPU / MPS / CUDA) with standard TRL + PEFT.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import List

from coliseum_defender.reward.mesa_reward import coliseum_grpo_reward_fn
from environment import TrustSafetyAuditEnv
from live_arena import LocalDefenderModel, defender_policy  # triggers load_dotenv()
from models import AuditAction

# ── After dotenv loads: lock down attacker mode, keep victim as configured ────
# Attacker: always local models (Qwen2.5-0.5B + DAN/Wild adapters).
# Victim:   whatever .env sets (groq by default) — user wants real Groq calls.
# Live API attacker: always off.
os.environ["USE_LIVE_ATTACKER_API"] = "0"
os.environ["USE_LOCAL_ATTACKER_MODELS"] = "1"
# VICTIM_PROVIDER intentionally NOT overridden — .env value (groq) is used.


TRAJECTORY_PATH = Path("data/live_grpo_trajectories.jsonl")

LOCAL_BASE   = Path(__file__).parent / "models" / "base" / "Qwen2.5-1.5B-Instruct"
LOCAL_SFT    = Path(__file__).parent / "models" / "adapters" / "coliseum-defender-sft"
MERGED_CACHE = Path(__file__).parent / "models" / "base" / "Qwen2.5-1.5B-sft-merged"


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def collect(episodes: int, turns: int, seed: int) -> None:
    # Lock down: only local model-generated adversarial attacks, no scripted/benign.
    os.environ["USE_LIVE_ATTACKER_API"] = "0"
    os.environ["USE_LOCAL_ATTACKER_MODELS"] = "1"
    os.environ["COLLECT_ADVERSARIAL_ONLY"] = "1"   # no scripted benign prompts

    TRAJECTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    victim_provider = os.getenv("VICTIM_PROVIDER", "groq")
    print(f"Collect mode: {episodes} episodes × {turns} turns")
    print("  Attackers : local Qwen2.5-0.5B + DAN/Wild adapters (NO scripted templates)")
    print("  Strategies: dan / wild / mutation only — 100% model-generated UNSAFE prompts")
    print("  Defender  : local Qwen2.5-1.5B + coliseum-defender-sft (MPS/CPU)")
    print(f"  Victim    : {victim_provider} (Groq API — llama-3.1-8b-instant, called on ALLOW)")
    print(f"  Output    : {TRAJECTORY_PATH}\n")

    # Load the local defender model once (Qwen2.5-1.5B + SFT adapter).
    defender_model = LocalDefenderModel()
    print("Loading local defender model (Qwen2.5-1.5B + SFT adapter)...")
    defender_model.load()
    print(f"Defender ready | backend={defender_model.backend}\n")

    env = TrustSafetyAuditEnv()
    total_rows = 0

    for ep in range(episodes):
        obs = env.reset(seed=seed + ep, episode_id=f"grpo_collect_{seed + ep}", total_turns=turns)
        done = False
        turn_num = 0
        ep_score = 0.0
        print(f"Episode {ep + 1}/{episodes} | seed={seed + ep}", flush=True)

        while not done:
            pred = defender_model.predict(obs)
            obs = env.step(AuditAction(decision=pred["decision"], reasoning=str(pred["reason"])))

            if pred["decision"].value.startswith("TOOL_"):
                # After tool call, run defender once more to get a final BLOCK/ALLOW
                pred2 = defender_model.predict(obs)
                if pred2["decision"].value.startswith("TOOL_"):
                    pred2 = defender_policy(obs)  # heuristic fallback to avoid infinite loop
                obs = env.step(AuditAction(decision=pred2["decision"], reasoning=str(pred2["reason"])))
                pred = pred2

            done = bool(obs.done)
            turn_num += 1
            total_rows += 1
            last = obs.conversation_history[-1] if obs.conversation_history else {}
            ep_score += float(last.get("reward", 0.0))
            print(
                f"  [{turn_num:02d}/{turns}] {pred['decision'].value:<5} "
                f"tier={obs.metadata.get('attack_tier')} "
                f"strategy={obs.metadata.get('attacker_strategy'):<8} "
                f"reward={last.get('reward', '?')}",
                flush=True,
            )

        print(f"  → episode done | ep_score={ep_score:.3f} | rows_so_far={total_rows}\n")

    print(f"Collection complete: {total_rows} trajectories written to {TRAJECTORY_PATH}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_prompts(path: Path) -> List[str]:
    prompts: list[str] = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row.get("prompt")
            label = row.get("true_label", "UNSAFE")
            if prompt:
                prompts.append(
                    "Review this user message for a trust and safety gate.\n"
                    "Return exactly:\n"
                    "Decision: BLOCK|ALLOW\nReason: <short reason>\nConfidence: <0-1>\n\n"
                    f"Message:\n{prompt}\n\nGround truth for verifier: {label}"
                )
    return prompts


def _plot_training_metrics(trainer, output_dir: str, max_steps: int, result) -> None:
    """Print a table of logged metrics and save a reward curve PNG."""
    log_history = getattr(getattr(trainer, "state", None), "log_history", [])

    steps, rewards, losses = [], [], []
    for entry in log_history:
        # TRL GRPOTrainer logs "reward" (aggregate mean), not "rewards/mean"
        if "reward" in entry and "step" in entry:
            steps.append(entry["step"])
            rewards.append(entry["reward"])
        if "loss" in entry and entry.get("loss") is not None and "step" in entry:
            losses.append((entry["step"], entry["loss"]))

    # ── Console table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRPO TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Final training loss : {result.training_loss:.4f}")
    print(f"  Steps completed     : {result.global_step}")
    if rewards:
        print(f"  Mean reward (start) : {rewards[0]:.4f}")
        print(f"  Mean reward (final) : {rewards[-1]:.4f}")
        print(f"  Reward delta        : {rewards[-1] - rewards[0]:+.4f}")
        print(f"  Best reward         : {max(rewards):.4f}")
        print(f"  Worst reward        : {min(rewards):.4f}")
    print("=" * 60)

    if not steps:
        print("No reward logs found — skipping curve.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2 if losses else 1, figsize=(14 if losses else 7, 5))
        if not losses:
            axes = [axes]

        # Reward curve
        ax = axes[0]
        ax.plot(steps, rewards, color="#3b82f6", linewidth=1.0, alpha=0.4, label="Raw reward")
        if len(rewards) >= 5:
            w = min(8, len(rewards))
            smooth = np.convolve(rewards, np.ones(w) / w, mode="valid")
            sx = steps[w // 2: w // 2 + len(smooth)]
            ax.plot(sx, smooth, color="#22c55e", linewidth=2.5, label=f"Smoothed (w={w})")
        ax.axhline(y=0, color="#ef4444", linestyle="--", alpha=0.5, label="Zero baseline")
        ax.set_title("GRPO Reward Curve", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Loss curve
        if losses:
            ax2 = axes[1]
            ls, lv = zip(*losses)
            ax2.plot(ls, lv, color="#f59e0b", linewidth=1.5)
            ax2.set_title("Training Loss", fontweight="bold")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.3)

        plt.suptitle(f"COLISEUM Defender GRPO — {max_steps} steps", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        curve_path = out / "grpo_reward_curve.png"
        plt.savefig(str(curve_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Reward curve saved → {curve_path}")

        # Also save raw metrics as JSON for later inspection
        metrics_path = out / "grpo_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "steps": steps,
                    "rewards_mean": rewards,
                    "loss_steps": [s for s, _ in losses],
                    "loss_values": [v for _, v in losses],
                    "final_loss": result.training_loss,
                    "global_step": result.global_step,
                },
                indent=2,
            )
        )
        print(f"Raw metrics saved  → {metrics_path}")

    except Exception as exc:
        print(f"Curve generation skipped: {exc}")


def _merge_sft_into_base(base_path: Path, sft_path: Path, merged_path: Path) -> None:
    """Merge SFT LoRA adapter into base weights and save a plain HF checkpoint."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Patch adapter_config so PEFT finds the local base model
    tmp = Path(str(sft_path) + "_merge_tmp")
    if tmp.exists():
        shutil.rmtree(str(tmp))
    shutil.copytree(str(sft_path), str(tmp))

    cfg_path = tmp / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    old_base = cfg.get("base_model_name_or_path", "")
    cfg["base_model_name_or_path"] = str(base_path)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  Patched base_model_name_or_path: {old_base!r} → {base_path}")

    print("  Loading base model on CPU (fp32)...")
    tok = AutoTokenizer.from_pretrained(str(base_path))
    mdl = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("  Applying SFT adapter and merging weights...")
    mdl = PeftModel.from_pretrained(mdl, str(tmp))
    mdl = mdl.merge_and_unload()
    mdl.save_pretrained(str(merged_path))
    tok.save_pretrained(str(merged_path))

    del mdl, tok
    gc.collect()

    shutil.rmtree(str(tmp), ignore_errors=True)
    print(f"  Merged model saved → {merged_path}")


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    sft_adapter: str,
    output_dir: str,
    max_steps: int,
    trajectory_path: Path,
    lora_r: int,
    lora_alpha: int,
    force_remerge: bool,
) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training requires datasets, transformers, peft, and trl. Install requirements.txt first."
        ) from exc

    prompts = _load_prompts(trajectory_path)
    if not prompts:
        raise SystemExit(
            f"No trajectories found at {trajectory_path}. "
            "Run `python train_grpo.py collect` first."
        )

    # ---- resolve paths ------------------------------------------------
    base_p   = Path(base_model)   if base_model   != "auto" else LOCAL_BASE
    sft_p    = Path(sft_adapter)  if sft_adapter  != "none" else LOCAL_SFT
    merged_p = MERGED_CACHE

    use_sft = sft_p.exists() and (sft_p / "adapter_config.json").exists()

    # ---- merge SFT into base (cached) ---------------------------------
    if use_sft:
        if force_remerge and merged_p.exists():
            shutil.rmtree(str(merged_p))
        if not merged_p.exists():
            print(f"Merging SFT adapter into base model → {merged_p} ...")
            merged_p.mkdir(parents=True, exist_ok=True)
            try:
                _merge_sft_into_base(base_p, sft_p, merged_p)
            except Exception as exc:
                shutil.rmtree(str(merged_p), ignore_errors=True)
                raise SystemExit(f"SFT merge failed: {exc}") from exc
        else:
            print(f"Using cached merged model: {merged_p}")
        load_from = str(merged_p)
    elif base_p.exists():
        print("No SFT adapter found — loading plain base model.")
        load_from = str(base_p)
    else:
        print(f"Local base not found, falling back to HF Hub: {base_model}")
        load_from = base_model

    device = _pick_device()
    print(f"\nDevice: {device}")

    # ---- load merged / base model ------------------------------------
    print(f"Loading model from: {load_from}")
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

    # ---- add fresh trainable LoRA for GRPO ---------------------------
    print(f"Adding fresh LoRA (r={lora_r}, alpha={lora_alpha}) for GRPO ...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- GRPO training -----------------------------------------------
    dataset = Dataset.from_dict({"prompt": prompts})

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,   # must equal num_generations (TRL requirement)
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=96,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=max(10, max_steps),
        fp16=(device == "cuda"),
        bf16=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=coliseum_grpo_reward_fn,
        args=grpo_args,
        train_dataset=dataset,
    )
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved GRPO adapter to: {output_dir}")

    # ---- metrics + reward curve ------------------------------------------
    _plot_training_metrics(trainer, output_dir, max_steps, result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect trajectories or train defender with GRPO.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("collect")
    cp.add_argument("--episodes", type=int, default=4)
    cp.add_argument("--turns",    type=int, default=24)
    cp.add_argument("--seed",     type=int, default=42)

    tp = sub.add_parser("train")
    tp.add_argument("--base_model",   default="auto",
                    help="Path to base model dir, or 'auto' to use models/base/Qwen2.5-1.5B-Instruct")
    tp.add_argument("--sft_adapter",  default="none",
                    help="Path to SFT adapter dir, or 'none' to skip merging")
    tp.add_argument("--output_dir",   default="models/adapters/coliseum-defender-grpo-live")
    tp.add_argument("--max_steps",    type=int, default=20)
    tp.add_argument("--lora_r",       type=int, default=16)
    tp.add_argument("--lora_alpha",   type=int, default=16)
    tp.add_argument("--trajectory_path", type=Path, default=TRAJECTORY_PATH)
    tp.add_argument("--force_remerge",   action="store_true",
                    help="Delete cached merged model and re-merge from scratch")

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.episodes, args.turns, args.seed)
    else:
        train(
            base_model=args.base_model,
            sft_adapter=args.sft_adapter,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            trajectory_path=args.trajectory_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            force_remerge=args.force_remerge,
        )


if __name__ == "__main__":
    main()
