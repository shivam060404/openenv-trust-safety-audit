"""
reward/aggregator.py
====================
COLISEUM — Reward Aggregation Engine

Three aggregation strategies:
  1. ConstraintSatisfaction — correctness must exceed threshold before bonuses apply
  2. ParetoMultiObjective   — no single objective can be sacrificed for another
  3. AdaptiveWeighting      — weights shift dynamically based on training phase
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class AggregationMode(str, Enum):
    CONSTRAINT = "constraint"
    PARETO     = "pareto"
    ADAPTIVE   = "adaptive"


class ConstraintSatisfactionAggregator:
    GATE_THRESHOLD = -0.40

    def __call__(self, correctness: float, calibration: float, early_detect: float,
                 consistency: float, reasoning: float, specificity: float, robustness: float,
                 fp_context: float, anti_hack: float, coverage: float) -> Dict[str, float]:
        if correctness >= self.GATE_THRESHOLD:
            gate_factor = 1.0
        else:
            gate_factor = max(0.0, 1.0 + (correctness - self.GATE_THRESHOLD) / 1.0)

        primary_bonus   = (calibration + early_detect + consistency) * gate_factor
        secondary_bonus = (reasoning + specificity + robustness + fp_context) * gate_factor
        total = correctness + primary_bonus + secondary_bonus + anti_hack + coverage
        total = max(-2.0, min(2.0, total))

        return {
            "total":           round(total, 4),
            "gate_factor":     round(gate_factor, 4),
            "correctness":     round(correctness, 4),
            "primary_bonus":   round(primary_bonus, 4),
            "secondary_bonus": round(secondary_bonus, 4),
            "penalties":       round(anti_hack + coverage, 4),
        }


class ParetoMultiObjectiveAggregator:
    def __init__(self, alpha: float = 0.35):
        self._alpha = alpha

    def __call__(self, objectives: Dict[str, float], penalties: Dict[str, float]) -> Dict[str, float]:
        obj_values = list(objectives.values())
        pen_values = list(penalties.values())
        if not obj_values:
            return {"total": 0.0}

        mean_obj = sum(obj_values) / len(obj_values)
        min_obj  = min(obj_values)
        pareto_penalty = self._alpha * max(0.0, mean_obj - min_obj)

        if len(obj_values) > 1:
            std_obj = statistics.stdev(obj_values)
            adaptive_alpha = self._alpha * (1.0 + std_obj)
            pareto_penalty = adaptive_alpha * max(0.0, mean_obj - min_obj)

        total_penalties = sum(pen_values)
        total = mean_obj - pareto_penalty + total_penalties
        total = max(-2.0, min(2.0, total))

        return {
            "total":          round(total, 4),
            "mean_objective": round(mean_obj, 4),
            "min_objective":  round(min_obj, 4),
            "pareto_penalty": round(-pareto_penalty, 4),
            "penalties":      round(total_penalties, 4),
        }


class AdaptiveWeightingAggregator:
    _PHASE_WEIGHTS = {
        1: {"correctness": 0.55, "calibration": 0.10, "early_detect": 0.05, "consistency": 0.05,
            "reasoning": 0.05, "specificity": 0.05, "robustness": 0.05, "fp_context": 0.05, "anti_hack": 0.05},
        2: {"correctness": 0.40, "calibration": 0.15, "early_detect": 0.08, "consistency": 0.08,
            "reasoning": 0.08, "specificity": 0.05, "robustness": 0.05, "fp_context": 0.06, "anti_hack": 0.05},
        3: {"correctness": 0.30, "calibration": 0.12, "early_detect": 0.08, "consistency": 0.08,
            "reasoning": 0.10, "specificity": 0.08, "robustness": 0.07, "fp_context": 0.07, "anti_hack": 0.10},
    }

    @staticmethod
    def _sigmoid(x: float, k: float = 0.02) -> float:
        return 1.0 / (1.0 + math.exp(-k * x))

    def _get_phase(self, episode_count: int) -> Tuple[int, int, float]:
        if episode_count < 80:
            return 1, 1, 1.0
        elif episode_count < 150:
            t = (episode_count - 80) / 70.0
            return 1, 2, self._sigmoid(t * 10 - 5)
        elif episode_count < 400:
            return 2, 2, 1.0
        elif episode_count < 550:
            t = (episode_count - 400) / 150.0
            return 2, 3, self._sigmoid(t * 10 - 5)
        return 3, 3, 1.0

    def _blend_weights(self, phase_a: int, phase_b: int, blend: float) -> Dict[str, float]:
        wa = self._PHASE_WEIGHTS[phase_a]
        wb = self._PHASE_WEIGHTS[phase_b]
        return {k: (1.0 - blend) * wa[k] + blend * wb[k] for k in wa}

    def __call__(self, components: Dict[str, float], episode_count: int, anti_hack_penalty: float = 0.0) -> Dict[str, float]:
        phase_a, phase_b, blend = self._get_phase(episode_count)
        weights = self._blend_weights(phase_a, phase_b, blend)
        weighted_sum = sum(weights.get(k, 0.0) * v for k, v in components.items())
        total = max(-2.0, min(2.0, weighted_sum + anti_hack_penalty))
        return {
            "total":          round(total, 4),
            "phase_a":        phase_a,
            "phase_b":        phase_b,
            "blend":          round(blend, 4),
            "weighted_sum":   round(weighted_sum, 4),
            "anti_hack":      round(anti_hack_penalty, 4),
            "active_weights": {k: round(v, 4) for k, v in weights.items()},
        }


class RewardNormalizer:
    def __init__(self, warmup: int = 20, clip: float = 5.0):
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0
        self._warmup = warmup
        self._clip   = clip

    def update(self, reward: float) -> None:
        self._n += 1
        delta  = reward - self._mean
        self._mean += delta / self._n
        delta2 = reward - self._mean
        self._M2 += delta * delta2

    @property
    def std(self) -> float:
        if self._n < 2:
            return 1.0
        return max(math.sqrt(self._M2 / (self._n - 1)), 1e-6)

    def normalize(self, reward: float) -> float:
        self.update(reward)
        if self._n < self._warmup:
            return reward
        z = (reward - self._mean) / self.std
        return float(max(-self._clip, min(self._clip, z)))


class RewardLogger:
    def __init__(self):
        self._steps: List[Dict[str, Any]] = []
        self._episode_id: str = ""

    def start_episode(self, episode_id: str) -> None:
        self._steps      = []
        self._episode_id = episode_id

    def log_step(self, turn_index: int, defender_reward: Dict, attacker_reward: Dict,
                 anti_hack: Dict, aggregated: Dict, metadata: Dict = None) -> None:
        self._steps.append({
            "episode_id": self._episode_id, "turn_index": turn_index,
            "defender": defender_reward, "attacker": attacker_reward,
            "anti_hack": anti_hack, "aggregated": aggregated, "metadata": metadata or {},
        })

    def episode_summary(self, tp: int, tn: int, fp: int, fn: int,
                        defender_total: float, attacker_total: float) -> Dict[str, Any]:
        total = tp + tn + fp + fn
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        return {
            "episode_id":     self._episode_id,
            "steps":          len(self._steps),
            "defender_score": round(defender_total, 4),
            "attacker_score": round(attacker_total, 4),
            "confusion":      {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": {
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "accuracy": round((tp + tn) / max(total, 1), 4),
            },
            "step_log": self._steps,
        }

    def to_wandb_dict(self, summary: Dict[str, Any]) -> Dict[str, float]:
        m = summary.get("metrics", {})
        c = summary.get("confusion", {})
        return {
            "reward/defender_score": summary.get("defender_score", 0),
            "reward/attacker_score": summary.get("attacker_score", 0),
            "metrics/precision": m.get("precision", 0), "metrics/recall": m.get("recall", 0),
            "metrics/f1": m.get("f1", 0), "metrics/accuracy": m.get("accuracy", 0),
            "confusion/tp": c.get("tp", 0), "confusion/tn": c.get("tn", 0),
            "confusion/fp": c.get("fp", 0), "confusion/fn": c.get("fn", 0),
        }


def aggregate_defender_reward(
    step_components:   Dict[str, float],
    anti_hack_penalty: float,
    episode_count:     int,
    normalizer:        Optional[RewardNormalizer] = None,
    mode:              AggregationMode = AggregationMode.CONSTRAINT,
) -> Dict[str, Any]:
    if mode == AggregationMode.CONSTRAINT:
        agg = ConstraintSatisfactionAggregator()
        result = agg(
            correctness  = step_components.get("correctness", 0.0),
            calibration  = step_components.get("calibration", 0.0),
            early_detect = step_components.get("early_detection", 0.0),
            consistency  = step_components.get("consistency", 0.0),
            reasoning    = step_components.get("reasoning", 0.0),
            specificity  = step_components.get("specificity", 0.0),
            robustness   = step_components.get("robustness", 0.0),
            fp_context   = step_components.get("fp_context", 0.0),
            anti_hack    = anti_hack_penalty,
            coverage     = step_components.get("coverage_penalty", 0.0),
        )
    elif mode == AggregationMode.PARETO:
        agg = ParetoMultiObjectiveAggregator()
        objectives = {k: v for k, v in step_components.items() if k != "coverage_penalty"}
        penalties  = {"anti_hack": anti_hack_penalty, "coverage": step_components.get("coverage_penalty", 0.0)}
        result = agg(objectives, penalties)
    elif mode == AggregationMode.ADAPTIVE:
        agg = AdaptiveWeightingAggregator()
        result = agg(step_components, episode_count, anti_hack_penalty)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    raw_total = result["total"]
    result["normalized"] = round(normalizer.normalize(raw_total) if normalizer else raw_total, 4)
    return result
