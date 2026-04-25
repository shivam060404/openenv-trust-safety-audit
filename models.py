"""
Pydantic models for the Trust & Safety Audit Environment.

Three required typed models per OpenEnv spec:
  - AuditAction:      What the agent submits each step
  - AuditObservation: What the agent receives each step
  - AuditReward:      Score model returned by graders

All models use Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from openenv.core import Action, Observation


# ── Decision Enum ─────────────────────────────────────────────────────────────

class Decision(str, Enum):
    """The agent's decision for each conversational turn, including tool use."""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    TOOL_POLICY_LOOKUP = "TOOL_POLICY_LOOKUP"
    TOOL_CHECK_CLEARANCE = "TOOL_CHECK_CLEARANCE"


# ── Action Model ──────────────────────────────────────────────────────────────

class AuditAction(Action):
    """
    Agent output submitted on every step() call.

    Attributes:
        decision: ALLOW, BLOCK, or a Tool call.
        reasoning: Optional explanation for logging/debugging. May be None.
    """
    decision: Decision = Field(
        ...,
        description="The agent's decision to ALLOW, BLOCK, or call a TOOL.",
    )
    reasoning: Optional[str] = Field(
        None,
        description="Optional reasoning for logging and debugging. May be None.",
    )


# ── Observation Model ─────────────────────────────────────────────────────────

class AuditObservation(Observation):
    """
    What the agent receives each step.

    Attributes:
        current_turn: The text of the current user prompt (attacker jailbreak).
        target_response: The target LLM (Groq) intended response to the prompt.
        session_id: Unique identifier for the current audit session/episode.
        turn_index: 0-based index of the current turn within the episode.
        total_turns: Total number of turns in this episode.
        conversation_history: Prior (turn, decision) pairs in this session.
        safety_ledger: The long-horizon tracking state.
        tool_output: If the last action was a tool call, the result is here.
        metadata: Source context — task_id, difficulty, attack_type, etc.
    """
    current_turn: str = Field(
        ...,
        description="The current user conversational turn (adversarial prompt) to be audited.",
    )
    target_response: str = Field(
        "",
        description="The Target LLM's response to the prompt.",
    )
    session_id: str = Field(
        ...,
        description="Unique identifier for the current audit session/episode.",
    )
    turn_index: int = Field(
        ...,
        description="0-based index of the current turn within the episode.",
    )
    total_turns: int = Field(
        ...,
        description="Total number of turns in this episode.",
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of prior (turn, decision, tool_output) pairs in this session.",
    )
    tool_output: Optional[str] = Field(
        None,
        description="Result of an external tool call if invoked in the previous step.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source context: task_id, difficulty, attack_tier if applicable.",
    )


# ── Reward Model ──────────────────────────────────────────────────────────────

class AuditReward(BaseModel):
    """
    Score model returned by graders. Required by openenv validate.

    Attributes:
        score: Final episode score clamped to [0.0, 1.0].
        raw_score: Unnormalised cumulative reward before clamping.
        max_possible_score: Maximum raw score achievable (for normalisation).
        false_positives: Benign turns incorrectly blocked.
        false_negatives: Malicious turns incorrectly allowed.
        turns_completed: Turns evaluated before episode ended.
    """
    score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Final episode score clamped to [0.0, 1.0].",
    )
    raw_score: float = Field(
        ...,
        description="Unnormalised cumulative reward before clamping. May be negative.",
    )
    max_possible_score: float = Field(
        ...,
        description="Maximum raw score achievable in this episode.",
    )
    false_positives: int = Field(
        0,
        description="Count of benign turns incorrectly blocked.",
    )
    false_negatives: int = Field(
        0,
        description="Count of malicious turns incorrectly allowed.",
    )
    turns_completed: int = Field(
        ...,
        description="Number of turns evaluated before episode ended.",
    )

    @property
    def normalised(self) -> float:
        """Normalise raw_score to [0.0, 1.0]."""
        if self.max_possible_score == 0:
            return 0.0
        return max(0.0, min(1.0, self.raw_score / self.max_possible_score))
