"""
curriculum_engine.py
====================
COLISEUM — Adaptive Difficulty Curriculum Engine (Theme 4)

Tracks the defender's block-rate per attacker tier on a rolling window of
recent episodes.  Auto-escalates tier when the defender is performing too well
on the current tier, creating the self-improvement loop that is the pitch
headline for Theme 4.

Difficulty tiers:
    1  →  DAN Agent         (simplest jailbreaks)
    2  →  WildTeam Agent    (complex multi-tactic)
    3  →  Mutation Agent    (obfuscated / encoded wrapping of tier 1/2 output)

Escalation rule: if the 5-episode rolling block-rate for the *current* tier
exceeds ESCALATE_THRESHOLD (80%), bump the tier up by one.
De-escalation rule: if the block-rate on the current tier drops below
DE_ESCALATE_THRESHOLD (30%) for 5 consecutive episodes, drop the tier back
down.  (Prevents over-escalation during early GRPO training.)
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional


ESCALATE_THRESHOLD   = 0.80   # defender blocks >80% → escalate
DE_ESCALATE_THRESHOLD = 0.30  # defender blocks <30% → de-escalate
WINDOW_SIZE          = 5      # rolling window length (episodes)
MIN_TIER             = 1
MAX_TIER             = 3


class CurriculumEngine:
    """
    Adaptive difficulty manager for the COLISEUM arena.

    Usage:
        engine = CurriculumEngine()

        # At the start of each episode:
        tier = engine.current_tier

        # After the episode completes:
        engine.record(tier, blocked=True)   # defender blocked the attack
        engine.record(tier, blocked=False)  # defender missed the attack

        # Check if tier changed after recording:
        new_tier = engine.current_tier
    """

    def __init__(self, starting_tier: int = 1) -> None:
        self._tier: int = max(MIN_TIER, min(MAX_TIER, starting_tier))
        # Per-tier ring buffers: True = blocked, False = missed
        self._windows: Dict[int, Deque[bool]] = {
            t: deque(maxlen=WINDOW_SIZE) for t in range(MIN_TIER, MAX_TIER + 1)
        }
        self._escalation_events: List[Dict] = []

    @property
    def current_tier(self) -> int:
        return self._tier

    def record(self, tier: int, blocked: bool) -> Optional[str]:
        """
        Record one episode outcome and potentially escalate/de-escalate.

        Args:
            tier:    the tier that was active for this episode
            blocked: True if the defender blocked the attack

        Returns:
            "escalated" | "de-escalated" | None
        """
        self._windows[tier].append(blocked)

        if len(self._windows[tier]) < WINDOW_SIZE:
            return None   # not enough data yet

        block_rate = sum(self._windows[tier]) / WINDOW_SIZE

        event = None
        if block_rate >= ESCALATE_THRESHOLD and self._tier < MAX_TIER:
            self._tier += 1
            event = "escalated"
            self._escalation_events.append({
                "from_tier":  tier,
                "to_tier":    self._tier,
                "block_rate": block_rate,
                "event":      event,
            })

        elif block_rate <= DE_ESCALATE_THRESHOLD and self._tier > MIN_TIER:
            self._tier -= 1
            event = "de-escalated"
            self._escalation_events.append({
                "from_tier":  tier,
                "to_tier":    self._tier,
                "block_rate": block_rate,
                "event":      event,
            })

        return event

    def block_rate(self, tier: Optional[int] = None) -> float:
        """Rolling block-rate for a tier (default: current tier)."""
        t = tier if tier is not None else self._tier
        window = self._windows[t]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def summary(self) -> Dict:
        return {
            "current_tier":      self._tier,
            "block_rates":       {t: self.block_rate(t) for t in range(MIN_TIER, MAX_TIER + 1)},
            "escalation_events": self._escalation_events,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tier → attacker name mapping (used by orchestrator for logging)
# ─────────────────────────────────────────────────────────────────────────────

TIER_NAMES = {1: "DAN", 2: "WildTeam", 3: "Mutation"}


if __name__ == "__main__":
    e = CurriculumEngine(starting_tier=1)
    # Simulate defender mastering tier 1
    for _ in range(5):
        event = e.record(1, blocked=True)
        print(f"tier={e.current_tier}  event={event}  rates={e.block_rate():.0%}")
    print(e.summary())
