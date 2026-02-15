from __future__ import annotations

from typing import Iterable

from ipo_risk_engine.report.schema import Action, HandState, RiskReport


def render_report(
    hand: HandState,
    *,
    action: Action,
    confidence: float,
    drivers: Iterable[str],
    assumptions: Iterable[str] | None = None,
) -> RiskReport:
    metrics = {
        **{k: float(v) for k, v in hand.features.items()},
        **{k: float(v) for k, v in hand.p_hat.items()},
    }

    return RiskReport(
        symbol=hand.symbol,
        asof=hand.asof,
        street=hand.street,
        action=action,
        confidence=float(confidence),
        drivers=list(drivers),
        metrics=metrics,
        assumptions=list(assumptions or []),
        comps=hand.comps,
    )
