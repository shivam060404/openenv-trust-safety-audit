"""
GRPO training entry point for the OpenEnv trust-safety arena.

This script supports two modes:
  1. collect: run the live environment and write GRPO-ready JSONL trajectories.
  2. train:   run TRL GRPOTrainer when model dependencies and compute are ready.

The live environment performs lightweight online adaptation during collection;
actual backpropagation happens here in batches/episodes, which is the practical
and stable path for hackathon compute and Hugging Face Spaces deployability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from coliseum_defender.reward.mesa_reward import coliseum_grpo_reward_fn
from environment import TrustSafetyAuditEnv
from live_arena import defender_policy
from models import AuditAction


TRAJECTORY_PATH = Path("data/live_grpo_trajectories.jsonl")


def collect(episodes: int, turns: int, seed: int) -> None:
    env = TrustSafetyAuditEnv()
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep, episode_id=f"grpo_collect_{seed + ep}", total_turns=turns)
        done = False
        while not done:
            pred = defender_policy(obs)
            obs = env.step(AuditAction(decision=pred["decision"], reasoning=str(pred["reason"])))
            if pred["decision"].value.startswith("TOOL_"):
                pred = defender_policy(obs)
                obs = env.step(AuditAction(decision=pred["decision"], reasoning=str(pred["reason"])))
            done = bool(obs.done)
    print(f"Wrote trajectories to {TRAJECTORY_PATH}")


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


def train(model_name: str, output_dir: str, max_steps: int, trajectory_path: Path) -> None:
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Training requires datasets, transformers, and trl. Install requirements.txt first."
        ) from exc

    prompts = _load_prompts(trajectory_path)
    if not prompts:
        raise SystemExit(f"No trajectories found at {trajectory_path}. Run `python train_grpo.py collect` first.")

    dataset = Dataset.from_dict({"prompt": prompts})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    args = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=768,
        max_completion_length=96,
        logging_steps=1,
        save_steps=max(10, max_steps),
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=coliseum_grpo_reward_fn,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GRPO adapter/model artifacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect trajectories or train defender with GRPO.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--episodes", type=int, default=4)
    collect_parser.add_argument("--turns", type=int, default=24)
    collect_parser.add_argument("--seed", type=int, default=42)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    train_parser.add_argument("--output_dir", default="models/adapters/coliseum-defender-grpo-live")
    train_parser.add_argument("--max_steps", type=int, default=20)
    train_parser.add_argument("--trajectory_path", type=Path, default=TRAJECTORY_PATH)

    args = parser.parse_args()
    if args.cmd == "collect":
        collect(args.episodes, args.turns, args.seed)
    else:
        train(args.model_name, args.output_dir, args.max_steps, args.trajectory_path)


if __name__ == "__main__":
    main()
