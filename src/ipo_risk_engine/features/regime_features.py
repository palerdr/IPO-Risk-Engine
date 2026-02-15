"""
Regime/context features: market-level signals computed over street windows.

These features capture what the broader market was doing while the IPO
was playing out — helping the model distinguish regime-dependent behavior
(e.g., a -30% MDD during a market crash vs. during a bull market).
"""
from __future__ import annotations

from datetime import date

import polars as pl

from ipo_risk_engine.features.streets import Street


# --- Sector → ETF mapping ---
SECTOR_ETF_MAP: dict[str, str] = {
    "technology": "XLK",
    "health_care": "XLV",
    "financials": "XLF",
    "industrials": "XLI",
    "consumer_goods": "XLP",
    "consumer_services": "XLY",
    "energy": "XLE",
    "oil_gas": "XLE",
    "real_estate": "XLRE",
    "airlines": "XLI",  # no pure airline ETF; industrials is closest
}


def get_sector_etf(sector: str) -> str | None:
    """Map a sector string to its proxy ETF ticker."""
    return SECTOR_ETF_MAP.get(sector)


def compute_regime_features(
    spy_bars: pl.DataFrame,
    sector_etf_bars: pl.DataFrame | None,
    street: Street,
    ipo_date: date,
) -> dict[str, float]:
    """
    Compute regime/context features for one street window.

    Args:
        spy_bars: Daily SPY bars already filtered to the street window.
        sector_etf_bars: Daily sector ETF bars filtered to the same window,
                         or None if no ETF mapping exists.
        street: Which street ("FLOP", "TURN", "RIVER") — used for feature naming.
        ipo_date: The IPO listing date — used for calendar features.

    Returns:
        Dict of feature_name -> value. Names are prefixed with street
        (e.g., "flop_spy_return", "turn_spy_vol").
    """
    prefix = street.lower()
    features: dict[str, float] = {}

    # --- SPY features ---
    spy_return, spy_vol = _compute_index_stats(spy_bars)
    features[f"{prefix}_spy_return"] = spy_return
    features[f"{prefix}_spy_vol"] = spy_vol

    # --- Sector ETF features ---
    if sector_etf_bars is not None and sector_etf_bars.height > 0:
        sector_return, _ = _compute_index_stats(sector_etf_bars)
        features[f"{prefix}_sector_return"] = sector_return
    else:
        features[f"{prefix}_sector_return"] = 0.0

    # Calendar features moved to PREFLOP (preflop_features.py)

    return features


def _compute_index_stats(bars: pl.DataFrame) -> tuple[float, float]:
    """
    Compute cumulative return and realized volatility from daily bars.

    Args:
        bars: Daily bars DataFrame with [ts, open, high, low, close, volume, ...].
              Already filtered to the relevant street window.

    Returns:
        (cumulative_return, realized_vol) tuple.
        - cumulative_return: (last_close - first_open) / first_open
        - realized_vol: std of daily log returns (0.0 if fewer than 2 bars)
    """
    if bars.height == 0:
        raise ValueError("No bars for this index")
    
    bars = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    )
    cumulative_return = (bars["close"][-1] - bars["open"][0]) / bars["open"][0]
    vol = bars["log_return"].std()
    realized_vol = float(vol) if vol is not None else 0.0
    return (cumulative_return, realized_vol)
