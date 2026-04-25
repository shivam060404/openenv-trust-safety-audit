"""
reward/reward_viz.py
====================
COLISEUM — Reward Visualization & Training Metrics

Tools for debugging and presenting the reward system during training.
Compatible with W&B, matplotlib, and terminal output.

Usage:
    from reward.reward_viz import RewardDashboard
    
    dashboard = RewardDashboard()
    dashboard.record(step_result)
    dashboard.print_summary()          # terminal table
    dashboard.plot_training_curves()   # matplotlib plots
    dashboard.export_wandb()           # W&B-compatible dict
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TO TRACK DURING TRAINING (Complete Specification)
# ─────────────────────────────────────────────────────────────────────────────

TRAINING_METRICS = {
    # === CORE PERFORMANCE ===
    "reward/total":              "Aggregated total reward per step (normalized)",
    "reward/correctness":        "TP/TN/FP/FN base reward with tier scaling",
    "reward/calibration":        "Brier-score-based confidence calibration",
    "reward/early_detection":    "Bonus for catching attacks at early turns",
    "reward/consistency":        "Intra-episode decision entropy score",
    "reward/reasoning":          "Structural reasoning quality score",
    "reward/specificity":        "Technique-naming Jaccard similarity",
    "reward/robustness":         "Extra credit for Tier-3 mutation handling",
    "reward/fp_context":         "Context-scaled false positive penalty",

    # === ANTI-HACK SIGNALS ===
    "anti_hack/always_block":    "Penalty for excessive blocking strategy",
    "anti_hack/always_allow":    "Penalty for passive allowing strategy",
    "anti_hack/grader_exploit":  "Penalty for keyword-correlation hacking",
    "anti_hack/entropy":         "Entropy out-of-range penalty",
    "anti_hack/pattern_mem":     "Memorization detection penalty",
    "anti_hack/diversity_audit": "Episode-level reasoning diversity penalty",
    "anti_hack/total":           "Sum of all anti-hack penalties this step",

    # === EPISODE METRICS ===
    "episode/tp_rate":           "True positive rate (recall) per episode",
    "episode/tn_rate":           "True negative rate (specificity) per episode",
    "episode/fp_rate":           "False positive rate per episode",
    "episode/fn_rate":           "False negative rate per episode",
    "episode/precision":         "Precision (TP / TP+FP)",
    "episode/recall":            "Recall (TP / TP+FN)",
    "episode/f1":                "F1 score per episode",
    "episode/coverage":          "Fraction of UNSAFE turns correctly blocked",
    "episode/total_reward":      "Cumulative reward across episode",

    # === CURRICULUM ===
    "curriculum/tier":           "Current attacker difficulty tier (1/2/3)",
    "curriculum/block_rate_t1":  "Rolling block rate on Tier-1 (DAN)",
    "curriculum/block_rate_t2":  "Rolling block rate on Tier-2 (WildTeam)",
    "curriculum/block_rate_t3":  "Rolling block rate on Tier-3 (Mutation)",
    "curriculum/escalations":    "Cumulative tier escalation events",

    # === ATTACKER (if training adversarially) ===
    "attacker/total_reward":     "Attacker total reward per step",
    "attacker/success_rate":     "Fraction of attacks that bypassed defender",
    "attacker/diversity":        "Shingle-based diversity score",
    "attacker/stealthiness":     "Delayed-detection stealthiness score",
    "attacker/mutation_eff":     "Weighted mutation technique effectiveness",

    # === NORMALIZATION ===
    "norm/reward_mean":          "Running mean of raw rewards (Welford)",
    "norm/reward_std":           "Running std of raw rewards (Welford)",
    "norm/raw_reward":           "Un-normalized reward before z-score",
}


# ─────────────────────────────────────────────────────────────────────────────
# RewardDashboard — tracks and visualizes all metrics
# ─────────────────────────────────────────────────────────────────────────────

class RewardDashboard:
    """
    Tracks per-step and per-episode reward metrics.
    Provides terminal visualization and W&B export.
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._history: List[Dict[str, Any]] = []
        self._rolling: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def record(self, step_result: Dict[str, Any]) -> None:
        """Record a single step result dict from compute_reward()."""
        self._history.append(step_result)

        # Flatten and record rolling stats
        breakdown = step_result.get("breakdown", {})
        anti_hack = breakdown.get("anti_hack", {})
        agg       = breakdown.get("aggregation", {})

        self._rolling["reward/total"].append(step_result.get("score", 0))
        self._rolling["reward/correctness"].append(breakdown.get("correctness", 0))
        self._rolling["reward/calibration"].append(breakdown.get("calibration", 0))
        self._rolling["reward/early_detection"].append(breakdown.get("early_detection", 0))
        self._rolling["reward/consistency"].append(breakdown.get("consistency", 0))
        self._rolling["reward/reasoning"].append(breakdown.get("reasoning", 0))
        self._rolling["reward/specificity"].append(breakdown.get("specificity", 0))
        self._rolling["reward/robustness"].append(breakdown.get("robustness", 0))
        self._rolling["anti_hack/total"].append(anti_hack.get("total_penalty",
            sum(v for k, v in anti_hack.items() if k != "total_penalty" and isinstance(v, (int, float)))))
        self._rolling["anti_hack/always_block"].append(anti_hack.get("always_block", 0))
        self._rolling["anti_hack/entropy"].append(anti_hack.get("entropy", 0))
        self._rolling["norm/raw_reward"].append(step_result.get("score", 0))

    def rolling_mean(self, key: str) -> float:
        """Rolling mean of a tracked metric."""
        data = self._rolling.get(key, [])
        if not data:
            return 0.0
        return sum(data) / len(data)

    def print_summary(self, last_n: int = 20) -> None:
        """Print a formatted terminal summary of recent metrics."""
        W = 70
        recent = self._history[-last_n:]
        if not recent:
            print("No steps recorded yet.")
            return

        print("\n" + "=" * W)
        print(f"  REWARD SYSTEM DASHBOARD  (last {len(recent)} steps)")
        print("=" * W)

        # Confusion matrix summary
        labels   = [s.get("label", "?") for s in recent]
        tp_count = labels.count("TP")
        tn_count = labels.count("TN")
        fp_count = labels.count("FP")
        fn_count = labels.count("FN")
        total    = len(labels)

        print(f"\n  Confusion Matrix (last {total} steps):")
        print(f"    TP={tp_count:>3} ({100*tp_count/max(total,1):.0f}%)  "
              f"TN={tn_count:>3} ({100*tn_count/max(total,1):.0f}%)  "
              f"FP={fp_count:>3} ({100*fp_count/max(total,1):.0f}%)  "
              f"FN={fn_count:>3} ({100*fn_count/max(total,1):.0f}%)")

        precision = tp_count / max(tp_count + fp_count, 1)
        recall    = tp_count / max(tp_count + fn_count, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-6)
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

        # Component breakdown
        print(f"\n  Rolling Averages (window={self._window}):")
        components = [
            ("reward/total",          "Total Reward"),
            ("reward/correctness",    "Correctness"),
            ("reward/calibration",    "Calibration"),
            ("reward/early_detection","Early Detection"),
            ("reward/consistency",    "Consistency"),
            ("reward/reasoning",      "Reasoning Quality"),
            ("reward/specificity",    "Tech Specificity"),
            ("reward/robustness",     "Adv Robustness"),
            ("anti_hack/total",       "Anti-Hack Penalty"),
            ("anti_hack/always_block","Always-Block Det."),
            ("anti_hack/entropy",     "Entropy Penalty"),
        ]

        for key, label in components:
            val  = self.rolling_mean(key)
            bar  = self._mini_bar(val, lo=-0.5, hi=1.0, width=20)
            sign = "+" if val >= 0 else ""
            print(f"    {label:<22}  {sign}{val:>+6.3f}  {bar}")

        print("=" * W + "\n")

    @staticmethod
    def _mini_bar(val: float, lo: float, hi: float, width: int = 20) -> str:
        """Render a mini ASCII bar chart for a value in [lo, hi]."""
        clamped = max(lo, min(hi, val))
        pos     = (clamped - lo) / max(hi - lo, 1e-6)
        filled  = int(pos * width)
        mid     = int((0 - lo) / max(hi - lo, 1e-6) * width)
        bar     = [" "] * width
        for i in range(min(mid, filled), max(mid, filled)):
            bar[i] = "█" if val >= 0 else "▒"
        bar[mid] = "│"
        return "[" + "".join(bar) + "]"

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Generate matplotlib training curve plots.
        Plots: total reward, component breakdown, confusion matrix, anti-hack penalties.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[RewardDashboard] matplotlib not installed. Run: pip install matplotlib")
            return

        if not self._history:
            print("[RewardDashboard] No data to plot.")
            return

        steps = list(range(len(self._history)))

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("COLISEUM Reward System — Training Dashboard", fontsize=14, fontweight="bold")
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # --- Panel 1: Total Reward ---
        ax1 = fig.add_subplot(gs[0, 0])
        raw_rewards  = [s.get("score", 0) for s in self._history]
        smooth_n     = max(1, len(raw_rewards) // 20)
        smoothed     = [
            sum(raw_rewards[max(0, i-smooth_n):i+1]) / min(i+1, smooth_n+1)
            for i in range(len(raw_rewards))
        ]
        ax1.plot(steps, raw_rewards, alpha=0.25, color="steelblue", label="Raw")
        ax1.plot(steps, smoothed,    color="steelblue", linewidth=2, label=f"Smooth({smooth_n})")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_title("Total Reward per Step")
        ax1.set_xlabel("Step"); ax1.set_ylabel("Reward")
        ax1.legend(fontsize=8)

        # --- Panel 2: Component Breakdown ---
        ax2 = fig.add_subplot(gs[0, 1])
        components = {
            "Correctness":  [s.get("breakdown", {}).get("correctness", 0) for s in self._history],
            "Calibration":  [s.get("breakdown", {}).get("calibration", 0) for s in self._history],
            "Reasoning":    [s.get("breakdown", {}).get("reasoning", 0) for s in self._history],
            "Anti-Hack":    [s.get("breakdown", {}).get("anti_hack", {}).get("total_penalty", 0) if isinstance(s.get("breakdown", {}).get("anti_hack"), dict) else 0 for s in self._history],
        }
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
        for (name, vals), color in zip(components.items(), colors):
            smooth_vals = [sum(vals[max(0,i-smooth_n):i+1])/min(i+1,smooth_n+1) for i in range(len(vals))]
            ax2.plot(steps, smooth_vals, label=name, color=color, linewidth=1.5)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_title("Reward Components (Smoothed)")
        ax2.set_xlabel("Step"); ax2.set_ylabel("Component Value")
        ax2.legend(fontsize=8)

        # --- Panel 3: Confusion Matrix Rates (rolling) ---
        ax3 = fig.add_subplot(gs[0, 2])
        window = max(1, len(self._history) // 10)
        tp_r, fp_r, fn_r = [], [], []
        for i in range(len(self._history)):
            w_labels = [s.get("label","?") for s in self._history[max(0,i-window):i+1]]
            total_w  = max(len(w_labels), 1)
            tp_r.append(w_labels.count("TP") / total_w)
            fp_r.append(w_labels.count("FP") / total_w)
            fn_r.append(w_labels.count("FN") / total_w)
        ax3.plot(steps, tp_r, color="#2ecc71", label="TP rate", linewidth=1.5)
        ax3.plot(steps, fp_r, color="#f39c12", label="FP rate", linewidth=1.5)
        ax3.plot(steps, fn_r, color="#e74c3c", label="FN rate", linewidth=1.5)
        ax3.set_ylim(0, 1)
        ax3.set_title("Rolling Confusion Rates")
        ax3.set_xlabel("Step"); ax3.set_ylabel("Rate")
        ax3.legend(fontsize=8)

        # --- Panel 4: Anti-Hack Penalties ---
        ax4 = fig.add_subplot(gs[1, 0])
        ah_keys = ["always_block", "always_allow", "grader_exploit", "entropy", "pattern_mem"]
        ah_colors = ["#e74c3c", "#e67e22", "#9b59b6", "#1abc9c", "#3498db"]
        for key, color in zip(ah_keys, ah_colors):
            vals = [
                s.get("breakdown", {}).get("anti_hack", {}).get(key, 0)
                if isinstance(s.get("breakdown", {}).get("anti_hack"), dict) else 0
                for s in self._history
            ]
            smooth_vals = [sum(vals[max(0,i-smooth_n):i+1])/min(i+1,smooth_n+1) for i in range(len(vals))]
            ax4.plot(steps, smooth_vals, label=key.replace("_", " ").title(), color=color, linewidth=1.2)
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax4.set_title("Anti-Hack Penalties (Smoothed)")
        ax4.set_xlabel("Step"); ax4.set_ylabel("Penalty")
        ax4.legend(fontsize=7)

        # --- Panel 5: Rolling F1 Score ---
        ax5 = fig.add_subplot(gs[1, 1])
        f1_rolling = []
        for i in range(len(self._history)):
            w_labels = [s.get("label","?") for s in self._history[max(0,i-window):i+1]]
            tp = w_labels.count("TP"); fp = w_labels.count("FP"); fn = w_labels.count("FN")
            p  = tp / max(tp + fp, 1);  r  = tp / max(tp + fn, 1)
            f1_rolling.append(2 * p * r / max(p + r, 1e-6))
        ax5.plot(steps, f1_rolling, color="#2980b9", linewidth=2)
        ax5.axhline(0.70, color="green",  linestyle="--", linewidth=0.8, label="Target 0.70")
        ax5.axhline(0.82, color="gold",   linestyle="--", linewidth=0.8, label="Elite 0.82")
        ax5.set_ylim(0, 1)
        ax5.set_title("Rolling F1 Score")
        ax5.set_xlabel("Step"); ax5.set_ylabel("F1")
        ax5.legend(fontsize=8)

        # --- Panel 6: Reward Distribution ---
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.hist(raw_rewards, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax6.axvline(0,  color="red",   linestyle="--", linewidth=1.2, label="Zero")
        ax6.axvline(sum(raw_rewards)/max(len(raw_rewards),1), color="gold",
                    linestyle="--", linewidth=1.2, label=f"Mean={sum(raw_rewards)/max(len(raw_rewards),1):.3f}")
        ax6.set_title("Reward Distribution")
        ax6.set_xlabel("Reward"); ax6.set_ylabel("Count")
        ax6.legend(fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[RewardDashboard] Saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()

    def export_wandb(self, step: int) -> Dict[str, float]:
        """Export current rolling stats as a flat W&B-compatible dict."""
        return {
            "reward/total":          self.rolling_mean("reward/total"),
            "reward/correctness":    self.rolling_mean("reward/correctness"),
            "reward/calibration":    self.rolling_mean("reward/calibration"),
            "reward/early_detection":self.rolling_mean("reward/early_detection"),
            "reward/consistency":    self.rolling_mean("reward/consistency"),
            "reward/reasoning":      self.rolling_mean("reward/reasoning"),
            "reward/specificity":    self.rolling_mean("reward/specificity"),
            "reward/robustness":     self.rolling_mean("reward/robustness"),
            "anti_hack/total":       self.rolling_mean("anti_hack/total"),
            "anti_hack/always_block":self.rolling_mean("anti_hack/always_block"),
            "anti_hack/entropy":     self.rolling_mean("anti_hack/entropy"),
            "step":                  step,
        }

    def save_json(self, path: str) -> None:
        """Save complete step history as JSON for offline analysis."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2, default=str)
        print(f"[RewardDashboard] Saved {len(self._history)} steps → {path}")
