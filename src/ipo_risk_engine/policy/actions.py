"""
Policy layer: convert risk scores → poker-style actions.

Actions:
  FOLD      — high tail risk, stay out
  SMALL_BET — uncertain, small position
  SIZE_UP   — low tail risk, full position

Dual event labels:
  adverse_20  — |MDD_20d| >= 20%, used for ranking/training diagnostics
  severe_30   — |MDD_20d| >= 30%, hard safety constraint

Threshold optimization targets:
  false-safe rate on severe_30 <= max_false_safe_rate
  (i.e., among true severe events, how often did we say SIZE_UP?)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ActionThresholds:
    """Frozen threshold config for p_tail → action mapping."""
    fold_min: float = 0.60
    small_bet_min: float = 0.30


def assign_action(p_tail: float, thresholds: ActionThresholds | None = None) -> str:
    """Map a single p_tail to an action string."""
    t = thresholds or ActionThresholds()
    if p_tail >= t.fold_min:
        return "FOLD"
    elif p_tail >= t.small_bet_min:
        return "SMALL_BET"
    else:
        return "SIZE_UP"


def assign_actions(p_tails: np.ndarray, thresholds: ActionThresholds | None = None) -> list[str]:
    """Map an array of p_tail values to actions."""
    return [assign_action(float(p), thresholds) for p in p_tails]


def summarize_actions(actions: list[str]) -> dict[str, int]:
    """Count occurrences of each action."""
    counts: dict[str, int] = {}
    for a in actions:
        counts[a] = counts.get(a, 0) + 1
    return dict(sorted(counts.items()))


def false_safe_rate(
    actions: list[str],
    severe_labels: np.ndarray,
) -> float:
    """Among true severe events, fraction labeled SIZE_UP."""
    severe_mask = severe_labels.astype(bool)
    if severe_mask.sum() == 0:
        return 0.0
    severe_actions = [a for a, s in zip(actions, severe_mask) if s]
    return sum(1 for a in severe_actions if a == "SIZE_UP") / len(severe_actions)


def coverage_report(
    actions: list[str],
    severe_labels: np.ndarray,
    adverse_labels: np.ndarray,
) -> dict[str, float]:
    """Compute key policy metrics."""
    n = len(actions)
    counts = summarize_actions(actions)
    fsr = false_safe_rate(actions, severe_labels)

    # Detection rate: what fraction of adverse events get FOLD?
    adverse_mask = adverse_labels.astype(bool)
    if adverse_mask.sum() > 0:
        adverse_actions = [a for a, m in zip(actions, adverse_mask) if m]
        detection_rate = sum(1 for a in adverse_actions if a == "FOLD") / len(adverse_actions)
    else:
        detection_rate = 0.0

    return {
        "n": n,
        "fold_pct": counts.get("FOLD", 0) / n,
        "small_bet_pct": counts.get("SMALL_BET", 0) / n,
        "size_up_pct": counts.get("SIZE_UP", 0) / n,
        "false_safe_rate": fsr,
        "adverse_detection_rate": detection_rate,
    }


def min_false_safe_rate(p_severe: float, c_sizeup: float) -> float:
    """Theoretical minimum false-safe rate for a given base rate and SIZE_UP fraction.

    Even with a perfect oracle ranking, if severe prevalence is high
    and SIZE_UP coverage is large, some severe events must leak through.

    Args:
        p_severe: Fraction of observations that are severe (e.g. 0.38 for tau=0.30).
        c_sizeup: Fraction of observations assigned SIZE_UP (e.g. 0.50).

    Returns:
        Best-case false-safe rate. If > max_false_safe_rate, the constraint
        is infeasible without reducing c_sizeup or raising the severity threshold.
    """
    if p_severe <= 0 or c_sizeup <= 0:
        return 0.0
    return max(0.0, (p_severe - (1 - c_sizeup)) / p_severe)


def feasibility_report(
    severe_labels: np.ndarray,
    max_false_safe_rate: float = 0.10,
) -> dict[str, float]:
    """Diagnose whether the safety constraint is feasible at various SIZE_UP levels.

    Returns a dict with p_severe, and for each candidate c_sizeup the
    theoretical min false-safe rate and whether the constraint is feasible.
    """
    p_severe = float(severe_labels.astype(bool).mean())
    result: dict[str, float] = {"p_severe": p_severe}

    for c_sizeup in [0.20, 0.30, 0.40, 0.50]:
        mfsr = min_false_safe_rate(p_severe, c_sizeup)
        result[f"min_fsr_at_{int(c_sizeup*100)}pct_sizeup"] = mfsr
        result[f"feasible_at_{int(c_sizeup*100)}pct_sizeup"] = float(mfsr <= max_false_safe_rate)

    max_sizeup = max(0.0, 1 - p_severe * (1 - max_false_safe_rate))
    result["max_feasible_sizeup"] = max_sizeup

    return result


def optimize_thresholds(
    p_tails: np.ndarray,
    severe_labels: np.ndarray,
    adverse_labels: np.ndarray,
    max_false_safe_rate: float = 0.10,
    fold_grid: np.ndarray | None = None,
    sbet_grid: np.ndarray | None = None,
) -> tuple[ActionThresholds, dict[str, float]]:
    """Grid search for thresholds that satisfy the safety constraint.

    Constraint: false_safe_rate <= max_false_safe_rate
    Objective: maximize SIZE_UP coverage (don't collapse to all-FOLD)

    Returns (best_thresholds, best_coverage_report).
    """
    if fold_grid is None:
        fold_grid = np.arange(0.30, 0.85, 0.05)
    if sbet_grid is None:
        sbet_grid = np.arange(0.10, 0.60, 0.05)

    best_thresholds = ActionThresholds()
    best_report = coverage_report(
        assign_actions(p_tails, best_thresholds), severe_labels, adverse_labels
    )
    best_size_up = best_report["size_up_pct"]

    for fold_min in fold_grid:
        for sbet_min in sbet_grid:
            if sbet_min >= fold_min:
                continue
            t = ActionThresholds(fold_min=float(fold_min), small_bet_min=float(sbet_min))
            actions = assign_actions(p_tails, t)
            report = coverage_report(actions, severe_labels, adverse_labels)

            if report["false_safe_rate"] > max_false_safe_rate:
                continue
            if report["size_up_pct"] > best_size_up:
                best_size_up = report["size_up_pct"]
                best_thresholds = t
                best_report = report

    return best_thresholds, best_report


def assign_actions_by_rank(
    scores: np.ndarray,
    fold_quantile: float = 0.75,
    sbet_quantile: float = 0.50,
) -> list[str]:
    """Assign actions based on score quantiles instead of calibrated probabilities.

    Use this when calibration is unreliable (< 50 severe events in train).
    Higher score = higher risk_severity = more dangerous.

    Args:
        scores: Raw committee scores (risk_severity predictions).
        fold_quantile: Scores above this percentile get FOLD.
        sbet_quantile: Scores between sbet and fold quantile get SMALL_BET.
    """
    fold_thresh = float(np.percentile(scores, fold_quantile * 100))
    sbet_thresh = float(np.percentile(scores, sbet_quantile * 100))
    actions = []
    for s in scores:
        if s >= fold_thresh:
            actions.append("FOLD")
        elif s >= sbet_thresh:
            actions.append("SMALL_BET")
        else:
            actions.append("SIZE_UP")
    return actions


MIN_SEVERE_EVENTS_FOR_CALIBRATION = 50


def should_calibrate(severe_labels: np.ndarray) -> bool:
    """Check if we have enough severe events to trust probability calibration."""
    return int(severe_labels.sum()) >= MIN_SEVERE_EVENTS_FOR_CALIBRATION


def optimize_thresholds_constrained(
    p_tails: np.ndarray,
    severe_labels: np.ndarray,
    adverse_labels: np.ndarray,
    max_false_safe_rate: float = 0.10,
    min_size_up_pct: float = 0.10,
    min_small_bet_pct: float = 0.0,
    max_fold_pct: float = 0.80,
    fold_grid: np.ndarray | None = None,
    sbet_grid: np.ndarray | None = None,
) -> tuple[ActionThresholds, dict[str, float]]:
    """Like optimize_thresholds but with coverage floor + fold ceiling.

    Prevents policy collapse to all-FOLD, all-SIZE_UP, or zero SMALL_BET.

    Constraints:
        false_safe_rate <= max_false_safe_rate
        size_up_pct >= min_size_up_pct
        small_bet_pct >= min_small_bet_pct
        fold_pct <= max_fold_pct
    """
    if fold_grid is None:
        fold_grid = np.arange(0.30, 0.85, 0.05)
    if sbet_grid is None:
        sbet_grid = np.arange(0.10, 0.60, 0.05)

    best_thresholds = ActionThresholds()
    best_report = coverage_report(
        assign_actions(p_tails, best_thresholds), severe_labels, adverse_labels
    )
    best_score = best_report["size_up_pct"]

    for fold_min in fold_grid:
        for sbet_min in sbet_grid:
            if sbet_min >= fold_min:
                continue
            t = ActionThresholds(fold_min=float(fold_min), small_bet_min=float(sbet_min))
            actions = assign_actions(p_tails, t)
            report = coverage_report(actions, severe_labels, adverse_labels)

            if report["false_safe_rate"] > max_false_safe_rate:
                continue
            if report["size_up_pct"] < min_size_up_pct:
                continue
            if report["small_bet_pct"] < min_small_bet_pct:
                continue
            if report["fold_pct"] > max_fold_pct:
                continue
            if report["size_up_pct"] > best_score:
                best_score = report["size_up_pct"]
                best_thresholds = t
                best_report = report

    return best_thresholds, best_report
