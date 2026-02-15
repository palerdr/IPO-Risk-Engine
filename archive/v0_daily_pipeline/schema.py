from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


Action = Literal["FOLD", "SMALL_BET", "SIZE_UP", "HEDGE", "EXIT"]
Street = Literal["PREFLOP", "FLOP", "TURN", "RIVER"]


@dataclass(frozen=True)
class HandState:
    """
    The state passed into your decision policy.

    - bars: the canonical bar dataframe restricted to data available up to `asof`
    - street: which information stage we're in
    - features: computed features for this street
    """
    symbol: str
    asof: datetime
    street: Street
    bars_1d: Any  # polars.DataFrame, but typed as Any to avoid importing polars everywhere
    features: dict[str, float]
    p_hat: dict[str, float]
    neighbor_diagnostics: dict[str, float]
    comps: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RiskReport:
    symbol: str
    asof: datetime
    street: Street

    action: Action
    confidence: float  # 0..1

    drivers: list[str]            # short bullet-like reasons
    metrics: dict[str, float]     # key numeric outputs
    assumptions: list[str]        # what you assumed / did not model
    comps: list[dict[str, Any]]
