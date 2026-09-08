"""
Street features: unified daily-bar feature computation for all streets.

All streets now use daily bars with street-specific prefixes:
  FLOP:  Days 0-5   (1d bars)
  TURN:  Days 6-20  (1d bars)
  RIVER: Days 21-60 (1d bars)
"""
from __future__ import annotations

import polars as pl
import numpy as np
from datetime import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipo_risk_engine.features.streets import Street

# Market hours in UTC (kept for reference, no longer used for filtering)
MARKET_OPEN_UTC = time(14, 30)   # 9:30 AM ET
FIRST_HOUR_END_UTC = time(15, 30)  # 10:30 AM ET
MARKET_CLOSE_UTC = time(21, 0)   # 4:00 PM ET

# Feature keys produced by compute_daily_street_features (unprefixed)
DAILY_FEATURE_KEYS = [
    "realized_vol",
    "range_mean",
    "dollar_volume_mean",
    "amihud_illiquidity",
    "cum_return",
    "trend_strength",
    "max_drawdown",
    "worst_day_return",
    "best_day_return",
    "volume_decay_ratio",
    "overnight_gap_mean",
]


def compute_daily_street_features(
    bars_1d: pl.DataFrame,
    street: Street,
) -> dict[str, float]:
    """Compute features from daily bars for any street.

    Feature names are prefixed with the lowercased street name:
      FLOP  -> flop_realized_vol, flop_range_mean, ...
      TURN  -> turn_realized_vol, turn_range_mean, ...
      RIVER -> river_realized_vol, river_range_mean, ...

    Args:
        bars_1d: Daily OHLCV bars filtered to the street window.
                 Must have columns [ts, open, high, low, close, volume].
        street: Which street (used for feature name prefix).

    Returns:
        Dict of prefixed feature names -> float values.
        If bars_1d has fewer than 2 rows, returns all None values.
    """
    prefix = street.lower()
    none_features = {f"{prefix}_{k}": None for k in DAILY_FEATURE_KEYS}

    if bars_1d.height < 2:
        return none_features

    bars = bars_1d.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        ((pl.col("open") - pl.col("close").shift(1)) / pl.col("close").shift(1)).alias("overnight_gap"),
    )

    realized_vol = float(bars["log_return"].std() or 0.0)
    range_mean = float(bars["range"].mean() or 0.0)
    dollar_volume_mean = float(bars["dollar_volume"].mean() or 0.0)

    # Guard against zero dollar_volume in illiquidity calc
    dv_safe = bars.filter(pl.col("dollar_volume") > 0)
    if dv_safe.height > 0:
        amihud_illiquidity = float((dv_safe["log_return"].abs() / dv_safe["dollar_volume"]).mean() or 0.0)
    else:
        amihud_illiquidity = 0.0

    cum_return = float((bars["close"][-1] - bars["open"][0]) / bars["open"][0])
    trend_strength = cum_return / realized_vol if realized_vol != 0 else 0.0

    max_drawdown = float(bars.select(
        ((pl.col("low") - pl.col("high").cum_max()) / pl.col("high").cum_max()).min()
    ).item())

    worst_day_return = float(bars["log_return"].min() or 0.0)
    best_day_return = float(bars["log_return"].max() or 0.0)

    # Volume decay: last day vs first day
    first_vol = bars["volume"][0]
    last_vol = bars["volume"][-1]
    volume_decay_ratio = float(last_vol / first_vol) if first_vol > 0 else 1.0

    overnight_gap_mean = float(bars["overnight_gap"].mean() or 0.0)

    raw = {
        "realized_vol": realized_vol,
        "range_mean": range_mean,
        "dollar_volume_mean": dollar_volume_mean,
        "amihud_illiquidity": amihud_illiquidity,
        "cum_return": cum_return,
        "trend_strength": trend_strength,
        "max_drawdown": max_drawdown,
        "worst_day_return": worst_day_return,
        "best_day_return": best_day_return,
        "volume_decay_ratio": volume_decay_ratio,
        "overnight_gap_mean": overnight_gap_mean,
    }
    return {f"{prefix}_{k}": v for k, v in raw.items()}


# ── DEPRECATED ─────────────────────────────────────────────────────
# The functions below used 5m/1h bars. Kept for reference only.
# All streets now use compute_daily_street_features() with 1d bars.
# ───────────────────────────────────────────────────────────────────

def compute_flop_features(bars_5m: pl.DataFrame) -> dict[str, float]:
    """
    Compute FLOP features from Day 0 5-minute bars.

    Features capture opening dynamics, first-hour behavior, and intraday patterns.

    Args:
        bars_5m: DataFrame with columns [ts, open, high, low, close, volume, vwap, trade_count]
                 Should contain only Day 0 bars.

    Returns:
        Dict of feature_name -> value
    """
    # Filter to market hours only
    bars = bars_5m.filter(
        (pl.col("ts").dt.time() >= MARKET_OPEN_UTC) &
        (pl.col("ts").dt.time() < MARKET_CLOSE_UTC)
    )

    if bars.height == 0:
        raise ValueError("No bars within market hours")

    # Split into first hour vs rest of day
    first_hour = bars.filter(pl.col("ts").dt.time() < FIRST_HOUR_END_UTC)
    rest_of_day = bars.filter(pl.col("ts").dt.time() >= FIRST_HOUR_END_UTC)

    # Key price extraction
    day_open = bars["open"][0]
    day_close = bars["close"][-1]
    day_high = bars["high"].max()
    day_high_ts = bars.filter(pl.col("high") == day_high).select("ts")[0, 0]
    day_low = bars["low"].min()
    day_vwap = bars["vwap"].mean()

    # Handle edge case: no bars in first hour
    if first_hour.height == 0:
        first_hour_close = day_open
    else:
        first_hour_close = first_hour["close"][-1]

    # Features
    features = {
        "first_hour_return": (first_hour_close - day_open) / day_open,
        "rest_of_day_return": (day_close - first_hour_close) / first_hour_close if first_hour_close != 0 else 0.0,
        "volume_first_hour_pct": first_hour["volume"].sum() / bars["volume"].sum() if first_hour.height > 0 else 0.0,
        "avg_trade_size": bars["volume"].sum() / bars["trade_count"].sum(),
        "close_vs_vwap": (day_close - day_vwap) / day_vwap,
        "time_to_high_minutes": (day_high_ts - bars["ts"][0]).total_seconds() / 60,
        "intraday_mdd": bars.select(
            ((pl.col("low") - pl.col("high").cum_max()) / pl.col("high").cum_max()).min()
        ).item(),
    }

    return features


def compute_turn_features(bars_1h: pl.DataFrame) -> dict[str, float]:
    """
    Compute TURN features from Days 1-5 hourly bars.

    Features capture overnight gaps, volume decay, and multi-day patterns.

    Args:
        bars_1h: DataFrame with columns [ts, open, high, low, close, volume, ...]
                 Should contain only Days 1-5 bars.

    Returns:
        Dict of feature_name -> value
    """
    # Filter to market hours
    bars = bars_1h.filter(
        (pl.col("ts").dt.time() >= MARKET_OPEN_UTC) &
        (pl.col("ts").dt.time() < MARKET_CLOSE_UTC)
    )

    if bars.height == 0:
        raise ValueError("No bars within market hours")

    # Compute hourly log returns for realized vol
    bars = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return")
    )

    # Aggregate to daily level
    daily = bars.group_by(pl.col("ts").dt.date().alias("date")).agg([
        pl.col("open").first().alias("day_open"),
        pl.col("close").last().alias("day_close"),
        pl.col("high").max().alias("day_high"),
        pl.col("low").min().alias("day_low"),
        pl.col("volume").sum().alias("day_volume"),
    ]).sort("date")

    # Add previous close for overnight gap calculation
    daily = daily.with_columns(
        pl.col("day_close").shift(1).alias("prev_close"),
    )

    # Compute overnight gaps
    daily = daily.with_columns(
        ((pl.col("day_open") - pl.col("prev_close")) / pl.col("prev_close")).alias("overnight_gap"),
        (pl.col("day_open") != pl.col("prev_close")).alias("is_gap"),
    )

    # Check if gap was filled (price revisited prev_close during the day)
    daily = daily.with_columns(
        ((pl.col("day_low") <= pl.col("prev_close")) &
         (pl.col("prev_close") <= pl.col("day_high"))
        ).alias("gap_filled")
    )

    # Compute gap fill ratio (with guard for no gaps)
    gaps_df = daily.filter(pl.col("is_gap") & pl.col("prev_close").is_not_null())
    if gaps_df.height > 0:
        gap_fill_ratio = gaps_df["gap_filled"].mean()
    else:
        gap_fill_ratio = 0.0

    # Volume decay: last day vs first day
    if daily.height >= 2:
        volume_decay_ratio = daily["day_volume"][-1] / daily["day_volume"][0]
    else:
        volume_decay_ratio = 1.0

    # Features
    features = {
        "overnight_gap_mean": daily["overnight_gap"].mean() or 0.0,
        "gap_fill_ratio": gap_fill_ratio,
        "volume_decay_ratio": volume_decay_ratio,
        "hourly_realized_vol": bars["log_return"].std() or 0.0,
        "turn_cum_return": (bars["close"][-1] - bars["open"][0]) / bars["open"][0],
        "max_drawdown_turn": bars.select(
            ((pl.col("low") - pl.col("high").cum_max()) / pl.col("high").cum_max()).min()
        ).item(),
    }

    return features


def compute_river_features(bars_1d: pl.DataFrame) -> dict[str, float]:
    """
    Compute RIVER features from Days 6-20 daily bars.

    Features capture longer-term trend, volatility regime, and stabilization.

    Args:
        bars_1d: DataFrame with columns [ts, open, high, low, close, volume, ...]
                 Should contain only Days 6-20 bars.

    Returns:
        Dict of feature_name -> value
    """
    #bars are already daily

    bars = bars_1d
    if bars.height == 0:
        raise ValueError("No bars within market hours")

    bars = bars.with_columns(
        (pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_return"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume")

    )
    realized_vol = bars["log_return"].std()
    range_mean = bars["range"].mean()
    dollar_volume_mean = bars["dollar_volume"].mean()
    amihud_illiquidity = (bars["log_return"].abs() / bars["dollar_volume"]).mean()
    cum_return = (bars["close"][-1] - bars["open"][0]) / bars["open"][0]
    trend_strength = ((bars["close"][-1] - bars["open"][0])/bars["open"][0])/bars["log_return"].std() if bars["log_return"].std() != 0 else 0
    max_drawdown_river = bars.select(
            ((pl.col("low") - pl.col("high").cum_max()) / pl.col("high").cum_max()).min()
        ).item()
    worst_day_return = bars["log_return"].min()
    best_day_return = bars["log_return"].max()


    features = {
        "realized_vol" : realized_vol,
        "range_mean" : range_mean,
        "dollar_volume_mean" : dollar_volume_mean,
        "amihud_illiquidity" : amihud_illiquidity,
        "river_cum_return" : cum_return,
        "trend_strength" : trend_strength,
        "max_drawdown_river" : max_drawdown_river,
        "worst_day_return" : worst_day_return,
        "best_day_return" : best_day_return,
    }
    return features