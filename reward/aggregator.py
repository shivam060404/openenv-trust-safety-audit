"""
reward/aggregator.py
====================
COLISEUM — Reward Aggregation Engine

Implements THREE aggregation strategies (selectable at runtime):
  1. ConstraintSatisfaction — correctness must exceed threshold before bonuses apply
  2. ParetoMultiObjective   — no single objective can be sacrificed for another
  3. AdaptiveWeighting      — weights shift dynamically based on training phase

Why NOT simple weighted sum?
  total = sum(w_i * r_i) is gameable: maximize one large component,
  ignore all others. A model will sacrifice calibration for correctness,
  ignore consistency, and forget reasoning quality entirely.

Constraint-based and Pareto-style aggregation prevent this by imposing
structural requirements that cannot be bypassed by single-component maximization.

Also provides:
  - RewardNormalizer: running z-score normalization for stable GRPO training
  - RewardLogger:     structured JSON-serializable logging for W&B/debugging
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation Mode Enum
# ─────────────────────────────────────────────────────────────────────────────

class AggregationMode(str, Enum):
    CONSTRAINT    = "constraint"       # Default — constraint satisfaction
    PARETO        = "pareto"           # Multi-objective Pareto-style
    ADAPTIVE      = "adaptive"         # Phase-aware adaptive weighting


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConstraintSatisfactionAggregator
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintSatisfactionAggregator:
    """
    Aggregates rewards with hard constraints on the correctness component.

    Architecture:
      - CORE reward:    correctness (gated — must exceed threshold)
      - PRIMARY bonuses: calibration, early detection, consistency
      - SECONDARY bonuses: reasoning, specificity, robustness
      - PENALTIES:      anti-hack, coverage (always applied)

    Gate mechanism:
      If correctness < GATE_THRESHOLD, bonuses are SCALED DOWN by gate_factor.
      This means: a model that makes the wrong decision barely benefits from
      well-formatted output or good calibration.

    Anti-hack:
      - Bonuses cannot compensate for wrong decisions
      - Model cannot sacrifice correctness to maximize secondary rewards
      - Penalties apply regardless of gate status
    """

    GATE_THRESHOLD = -0.40  # Correctness below this → bonuses suppressed

    def __call__(
        self,
        correctness:   float,
        calibration:   float,
        early_detect:  float,
        consistency:   float,
        reasoning:     float,
        specificity:   float,
        robustness:    float,
        fp_context:    float,
        anti_hack:     float,
        coverage:      float,
    ) -> Dict[str, float]:
        # Gate factor: 1.0 if correctness OK, drops toward 0 for very wrong decisions
        if correctness >= self.GATE_THRESHOLD:
            gate_factor = 1.0
        else:
            # Smooth transition: gate_factor → 0 as correctness → -1.5
            gate_factor = max(0.0, 1.0 + (correctness - self.GATE_THRESHOLD) / 1.0)

        primary_bonus   = (calibration + early_detect + consistency) * gate_factor
        secondary_bonus = (reasoning + specificity + robustness + fp_context) * gate_factor

        total = correctness + primary_bonus + secondary_bonus + anti_hack + coverage
        total = max(-2.0, min(2.0, total))

        return {
            "total":          round(total, 4),
            "gate_factor":    round(gate_factor, 4),
            "correctness":    round(correctness, 4),
            "primary_bonus":  round(primary_bonus, 4),
            "secondary_bonus": round(secondary_bonus, 4),
            "penalties":      round(anti_hack + coverage, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. ParetoMultiObjectiveAggregator
# ─────────────────────────────────────────────────────────────────────────────

class ParetoMultiObjectiveAggregator:
    """
    Pareto-style aggregation: penalizes extreme under-performance in ANY objective.

    For N objectives, we compute:
      1. Mean reward across objectives (baseline)
      2. Minimum reward (weakest objective)
      3. Final = mean - α * max(0, mean - minimum)   [α is Pareto penalty weight]

    Effect: if ANY objective is very negative, the penalty term kicks in
    and drags down the total. Model cannot ignore any single dimension.

    This mimics Pareto efficiency: a solution is only good if it's not
    catastrophically bad in any direction.

    Anti-hack:
      - Uses minimum (not sum) as Pareto violation signal
      - Cannot be gamed by maximizing one term — minimum pulls everything down
      - α is adaptive based on variance of objectives
    """

    def __init__(self, alpha: float = 0.35):
        self._alpha = alpha

    def __call__(
        self,
        objectives: Dict[str, float],
        penalties:  Dict[str, float],
    ) -> Dict[str, float]:
        """
        Args:
            objectives: dict of {name: reward_value} for all positive objectives
            penalties:  dict of {name: penalty_value} for all anti-hack penalties
        """
        obj_values = list(objectives.values())
        pen_values = list(penalties.values())

        if not obj_values:
            return {"total": 0.0}

        mean_obj = sum(obj_values) / len(obj_values)
        min_obj  = min(obj_values)

        # Pareto penalty: proportional to how much the weakest objective drags
        pareto_penalty = self._alpha * max(0.0, mean_obj - min_obj)

        # Adaptive alpha: higher variance = higher Pareto enforcement
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


# ─────────────────────────────────────────────────────────────────────────────
# 3. AdaptiveWeightingAggregator
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveWeightingAggregator:
    """
    Phase-aware adaptive weighting that shifts emphasis as training progresses.

    Three training phases (determined by episode count):
      Phase 1 (0–100 eps):  Correctness + Format dominant → build basic competence
      Phase 2 (100–500 eps): Add calibration + reasoning → develop understanding
      Phase 3 (500+ eps):   Full multi-objective with anti-hack emphasis → generalize

    Phase transition uses sigmoid smoothing — no hard jumps that cause gradient spikes.

    Why this is robust:
      - Early training: simplified signal avoids premature exploration of hack strategies
      - Late training: full penalties enforce generalization when model is capable
      - Weights sum to 1.0 at all times (normalization-preserving)
    """

    # Weight tables per phase: {component: weight}
    _PHASE_WEIGHTS = {
        1: {  # Basic: correctness dominant
            "correctness":   0.55,
            "calibration":   0.10,
            "early_detect":  0.05,
            "consistency":   0.05,
            "reasoning":     0.05,
            "specificity":   0.05,
            "robustness":    0.05,
            "fp_context":    0.05,
            "anti_hack":     0.05,
        },
        2: {  # Developing: balanced correctness + understanding
            "correctness":   0.40,
            "calibration":   0.15,
            "early_detect":  0.08,
            "consistency":   0.08,
            "reasoning":     0.08,
            "specificity":   0.05,
            "robustness":    0.05,
            "fp_context":    0.06,
            "anti_hack":     0.05,
        },
        3: {  # Mature: full multi-objective
            "correctness":   0.30,
            "calibration":   0.12,
            "early_detect":  0.08,
            "consistency":   0.08,
            "reasoning":     0.10,
            "specificity":   0.08,
            "robustness":    0.07,
            "fp_context":    0.07,
            "anti_hack":     0.10,
        },
    }

    @staticmethod
    def _sigmoid(x: float, k: float = 0.02) -> float:
        return 1.0 / (1.0 + math.exp(-k * x))

    def _get_phase(self, episode_count: int) -> Tuple[int, int, float]:
        """Returns (phase_a, phase_b, blend) for smooth transition."""
        if episode_count < 80:
            return 1, 1, 1.0
        elif episode_count < 150:
            t = (episode_count - 80) / 70.0
            blend = self._sigmoid(t * 10 - 5)
            return 1, 2, blend
        elif episode_count < 400:
            return 2, 2, 1.0
        elif episode_count < 550:
            t = (episode_count - 400) / 150.0
            blend = self._sigmoid(t * 10 - 5)
            return 2, 3, blend
        else:
            return 3, 3, 1.0

    def _blend_weights(self, phase_a: int, phase_b: int, blend: float) -> Dict[str, float]:
        wa = self._PHASE_WEIGHTS[phase_a]
        wb = self._PHASE_WEIGHTS[phase_b]
        return {
            k: (1.0 - blend) * wa[k] + blend * wb[k]
            for k in wa
        }

    def __call__(
        self,
        components: Dict[str, float],
        episode_count: int,
        anti_hack_penalty: float = 0.0,
    ) -> Dict[str, float]:
        phase_a, phase_b, blend = self._get_phase(episode_count)
        weights = self._blend_weights(phase_a, phase_b, blend)

        weighted_sum = sum(
            weights.get(k, 0.0) * v
            for k, v in components.items()
        )

        total = weighted_sum + anti_hack_penalty
        total = max(-2.0, min(2.0, total))

        return {
            "total":          round(total, 4),
            "phase_a":        phase_a,
            "phase_b":        phase_b,
            "blend":          round(blend, 4),
            "weighted_sum":   round(weighted_sum, 4),
            "anti_hack":      round(anti_hack_penalty, 4),
            "active_weights": {k: round(v, 4) for k, v in weights.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# RewardNormalizer — Z-score running normalization for GRPO stability
# ─────────────────────────────────────────────────────────────────────────────

class RewardNormalizer:
    """
    Running z-score normalization for GRPO training stability.

    GRPO is sensitive to reward scale. Raw rewards spanning [-2, +2]
    can cause gradient instability. This normalizer maintains a running
    mean and std, normalizing each reward before it's fed to the trainer.

    Uses Welford's online algorithm for numerically stable running stats.

    Warm-up: first 20 rewards use the raw value (normalization unstable with N<20)
    """

    def __init__(self, warmup: int = 20, clip: float = 5.0):
        self._n    = 0
        self._mean = 0.0
        self._M2   = 0.0   # sum of squared deviations (Welford)
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
            return reward  # raw during warmup
        z = (reward - self._mean) / self.std
        return float(max(-self._clip, min(self._clip, z)))


# ─────────────────────────────────────────────────────────────────────────────
# RewardLogger — structured logging for W&B / debugging
# ─────────────────────────────────────────────────────────────────────────────

class RewardLogger:
    """
    Captures every reward component for every step.
    Produces structured dicts compatible with wandb.log() and JSON serialization.

    Schema:
    {
        "episode_id":   str,
        "turn_index":   int,
        "step_rewards": {
            "defender": {
                "total": float,
                "breakdown": {...},
                "anti_hack": {...},
                "aggregated": {...}
            },
            "attacker": {
                "total": float,
                "breakdown": {...}
            }
        },
        "episode_totals": {
            "defender_score": float,
            "attacker_score": float,
            "tp": int, "tn": int, "fp": int, "fn": int,
            "precision": float, "recall": float, "f1": float
        }
    }
    """

    def __init__(self):
        self._steps: List[Dict[str, Any]] = []
        self._episode_id: str = ""

    def start_episode(self, episode_id: str) -> None:
        self._steps      = []
        self._episode_id = episode_id

    def log_step(
        self,
        turn_index:      int,
        defender_reward: Dict[str, Any],
        attacker_reward: Dict[str, Any],
        anti_hack:       Dict[str, Any],
        aggregated:      Dict[str, Any],
        metadata:        Dict[str, Any] = None,
    ) -> None:
        self._steps.append({
            "episode_id":  self._episode_id,
            "turn_index":  turn_index,
            "defender":    defender_reward,
            "attacker":    attacker_reward,
            "anti_hack":   anti_hack,
            "aggregated":  aggregated,
            "metadata":    metadata or {},
        })

    def episode_summary(
        self,
        tp: int, tn: int, fp: int, fn: int,
        defender_total: float,
        attacker_total: float,
    ) -> Dict[str, Any]:
        total = tp + tn + fp + fn
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy  = (tp + tn) / max(total, 1)

        summary = {
            "episode_id":     self._episode_id,
            "steps":          len(self._steps),
            "defender_score": round(defender_total, 4),
            "attacker_score": round(attacker_total, 4),
            "confusion":      {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "metrics": {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "accuracy":  round(accuracy, 4),
            },
            "step_log": self._steps,
        }
        return summary

    def to_wandb_dict(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Flat dict for wandb.log() compatibility."""
        m = summary.get("metrics", {})
        c = summary.get("confusion", {})
        return {
            "reward/defender_score":   summary.get("defender_score", 0),
            "reward/attacker_score":   summary.get("attacker_score", 0),
            "metrics/precision":       m.get("precision", 0),
            "metrics/recall":          m.get("recall", 0),
            "metrics/f1":              m.get("f1", 0),
            "metrics/accuracy":        m.get("accuracy", 0),
            "confusion/tp":            c.get("tp", 0),
            "confusion/tn":            c.get("tn", 0),
            "confusion/fp":            c.get("fp", 0),
            "confusion/fn":            c.get("fn", 0),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API: aggregate_defender_reward
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_defender_reward(
    step_components:  Dict[str, float],
    anti_hack_penalty: float,
    episode_count:    int,
    normalizer:       Optional[RewardNormalizer] = None,
    mode:             AggregationMode = AggregationMode.CONSTRAINT,
) -> Dict[str, Any]:
    """
    Main aggregation entry point for defender reward.

    Args:
        step_components:  {'correctness': f, 'calibration': f, ...}
        anti_hack_penalty: total penalty from anti_hack.py
        episode_count:     total episodes completed (for adaptive weighting)
        normalizer:        optional running normalizer (pass through from training loop)
        mode:              aggregation strategy

    Returns:
        {'total': float, 'normalized': float, 'details': dict}
    """
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
        penalties  = {"anti_hack": anti_hack_penalty,
                      "coverage":  step_components.get("coverage_penalty", 0.0)}
        result = agg(objectives, penalties)

    elif mode == AggregationMode.ADAPTIVE:
        agg = AdaptiveWeightingAggregator()
        result = agg(step_components, episode_count, anti_hack_penalty)

    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    # Normalize for GRPO
    raw_total = result["total"]
    if normalizer:
        normalized = normalizer.normalize(raw_total)
    else:
        normalized = raw_total

    result["normalized"] = round(normalized, 4)
    return result
