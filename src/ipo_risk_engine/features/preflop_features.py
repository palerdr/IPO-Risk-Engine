"""
PREFLOP features: structured metadata computed before market open on IPO day.

No bar data needed — uses EDGAR metadata, universe statistics, and
pre-IPO market context from SPY/sector ETF bars.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from ipo_risk_engine.features.regime_features import get_sector_etf


def compute_preflop_features(
    ipo_date: date,
    sic: str | None,
    filing_count: int,
    universe_df: pl.DataFrame,
    spy_bars_1d: pl.DataFrame | None,
    sector_etf_bars_1d: pl.DataFrame | None,
) -> dict[str, float]:
    """Compute PREFLOP features from EDGAR metadata + pre-IPO market context.

    All features use data strictly available before market open on ipo_date.

    Args:
        ipo_date: IPO listing date.
        sic: 4-digit SIC code (or None).
        filing_count: Number of S-1/F-1 filings for this CIK.
        universe_df: Full IPO universe DataFrame (for sector heat).
        spy_bars_1d: SPY daily bars (pre-fetched, full range).
        sector_etf_bars_1d: Sector ETF daily bars (or None).

    Returns:
        Dict of "preflop_*" feature names -> float values.
    """
    features: dict[str, float] = {}

    # -- EDGAR metadata --
    features["preflop_filing_count"] = float(filing_count)
    features["preflop_ipo_month"] = float(ipo_date.month)
    features["preflop_ipo_day_of_week"] = float(ipo_date.weekday())

    # -- Sector IPO heat: count of IPOs in same sector within 90 days before --
    features["preflop_sector_ipo_heat_90d"] = _sector_heat(
        ipo_date, sic, universe_df
    )

    # -- Pre-IPO market context (20 trading days before IPO) --
    spy_ret, spy_vol = _pre_ipo_market_stats(spy_bars_1d, ipo_date, n_days=20)
    features["preflop_spy_return_20d"] = spy_ret
    features["preflop_spy_vol_20d"] = spy_vol

    sector_ret = 0.0
    if sector_etf_bars_1d is not None:
        sector_ret, _ = _pre_ipo_market_stats(sector_etf_bars_1d, ipo_date, n_days=20)
    features["preflop_sector_return_20d"] = sector_ret

    return features


def _sector_heat(
    ipo_date: date,
    sic: str | None,
    universe_df: pl.DataFrame,
) -> float:
    """Count IPOs in the same 2-digit SIC sector within 90 days before ipo_date."""
    if not sic or len(sic) < 2:
        return 0.0

    sic_2 = sic[:2]

    # Filter universe to same 2-digit SIC, within 90 days before (exclusive of same day)
    heat = universe_df.filter(
        pl.col("sic").is_not_null()
        & pl.col("sic").str.starts_with(sic_2)
        & (pl.col("ipo_date") < ipo_date)
        & (pl.col("ipo_date") >= pl.lit(ipo_date).cast(pl.Date) - pl.duration(days=90))
    )
    return float(heat.height)


def _pre_ipo_market_stats(
    bars_1d: pl.DataFrame | None,
    ipo_date: date,
    n_days: int = 20,
) -> tuple[float, float]:
    """Compute cumulative return and realized vol from the n trading days before ipo_date."""
    if bars_1d is None or bars_1d.height == 0:
        return 0.0, 0.0

    # Filter to bars strictly before IPO date
    pre = bars_1d.filter(pl.col("ts").dt.date() < ipo_date).sort("ts").tail(n_days)

    if pre.height < 2:
        return 0.0, 0.0

    cum_return = float((pre["close"][-1] - pre["open"][0]) / pre["open"][0])
    pre = pre.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    )
    vol = pre["log_return"].std()
    realized_vol = float(vol) if vol is not None else 0.0

    return cum_return, realized_vol
