"""
Lightweight online adaptation utilities for live Coliseum sessions.

This module does not pretend to replace GRPO. It records GRPO-ready trajectory
data and applies safe online bandit updates to sampling weights during the live
demo. Full backpropagation is handled by train_grpo.py / TRL using this data.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


LIVE_LOG_PATH = Path("live_interactive_logs.log")
TRAJECTORY_PATH = Path("data/live_grpo_trajectories.jsonl")


def initialize_live_log(path: Path = LIVE_LOG_PATH) -> None:
    """Start a clean JSONL log file for an explicit live demo run."""
    path.write_text("", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def log_live_event(event: str, payload: Dict[str, Any]) -> None:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **payload,
    }
    append_jsonl(LIVE_LOG_PATH, record)


@dataclass
class AdaptiveBandit:
    """Small softmax bandit for choosing attacker strategies per session."""

    arms: Iterable[str]
    learning_rate: float = 0.08
    temperature: float = 0.8
    weights: Dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.weights = {arm: 0.0 for arm in self.arms}

    def choose(self, rng) -> str:
        names = list(self.weights)
        logits = [self.weights[name] / max(0.05, self.temperature) for name in names]
        max_logit = max(logits)
        probs = [math.exp(logit - max_logit) for logit in logits]
        total = sum(probs)
        threshold = rng.random() * total
        upto = 0.0
        for name, prob in zip(names, probs):
            upto += prob
            if upto >= threshold:
                return name
        return names[-1]

    def update(self, arm: str, reward: float) -> None:
        if arm in self.weights:
            self.weights[arm] += self.learning_rate * reward

    def summary(self) -> Dict[str, float]:
        return {key: round(value, 4) for key, value in sorted(self.weights.items())}


@dataclass
class ExperienceBuffer:
    """Records attacker, defender, and environment outcomes for GRPO batches."""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        append_jsonl(TRAJECTORY_PATH, row)

    def grouped_rewards(self) -> Dict[str, float]:
        totals: Dict[str, list[float]] = {}
        for row in self.rows:
            strategy = str(row.get("attacker_strategy", "unknown"))
            totals.setdefault(strategy, []).append(float(row.get("attacker_reward", 0.0)))
        return {
            strategy: round(sum(vals) / max(1, len(vals)), 4)
            for strategy, vals in sorted(totals.items())
        }
