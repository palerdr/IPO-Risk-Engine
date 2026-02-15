from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import polars as pl


@dataclass(frozen=True)
class WindowedBars:
    symbol: str
    start: datetime
    end: datetime  # [start, end)
    bars: pl.DataFrame  # canonical bars_1d slice


def slice_window(bars_1d: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    #Validate required columns exist to prevent silent failures later when you run features/policy/backtests.
    required = {"symbol", "ts", "open", "high", "low", "close", "volume"}
    missing = required - set(bars_1d.columns)
    if missing:
        raise ValueError(f"bars_1d missing required columns: {sorted(missing)}")

    #Filter by the window boundaries using Polars expressions.
    window = bars_1d.filter(
        (pl.col("ts") >= start) & (pl.col("ts") < end)
    )

    #Ensure the slice is sorted by time.
    window = window.sort(["symbol", "ts"])

    return window



def add_basic_returns(bars: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a new dataframe with basic return columns added.

    Intended columns (names are part of your internal contract):
    - ret_1d: close-to-close simple return
    - logret_1d: log(1+ret_1d)
    - gap_oc: open-to-prev-close return (overnight gap proxy)
    - intraday_ret: close-to-open return

    Tools you'll use (Polars):
    - pl.col(...).shift(1)
    - (a / b - 1)
    - pl.log1p(...)
    """
    required = {"ts", "open", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns for returns: {sorted(missing)}")
    prev_close = pl.col("close").shift(1)
    out = bars.with_columns(
        [
            (pl.col("close")/prev_close - 1.0).alias("ret_1d"),
            (pl.col("close")/prev_close - 1.0).log1p().alias("logret_1d"),
            (pl.col("open")/prev_close - 1.0).alias("gap_oc"),
            (pl.col("close")/pl.col("open") - 1.0).alias("intraday_ret"),

        ]
    )
     
    return out

def compute_core_features(window: pl.DataFrame) -> dict[str, float]:
    """
    Compute a small, stable feature set from a window of daily bars.

    MVP feature families (10–15 total):
    1) Vol / path
       - realized_vol (stdev of logret_1d)
       - range_mean: mean((high-low)/open)
       - max_drawdown_in_window (path-based)
    2) Liquidity proxies
       - dollar_volume_mean: mean(close * volume)
       - amihud_mean: mean(abs(ret_1d) / dollar_volume)
       - trade_count_mean (if present)
    3) Momentum / drift
       - cum_return (close_end / close_start - 1)
       - trend_strength: fraction of positive days
    4) Tail / jumpiness
       - worst_day_return (min ret_1d)
       - best_day_return (max ret_1d)

    Output should be pure floats (json-serializable).
    """
    if window.is_empty():
        return{
            "realized_vol": 0.0,
            "range_mean": 0.0,
            "cum_return": 0.0,
            "worst_day_return": 0.0,
            "best_day_return": 0.0,
            "trend_strength": 0.0,
            "dollar_volume_mean": 0.0,
            "amihud_mean": 0.0,
            "trade_count_mean": 0.0,
            "max_drawdown_in_window": 0.0,
        }
    df = add_basic_returns(window)
    dollar_volume = (pl.col("close")*pl.col("volume")).alias("dollar_volume")
    df = df.with_columns([dollar_volume])

    realized_vol = df.select(pl.col("logret_1d").std()).item()
    range_mean = df.select(((pl.col("high") - pl.col("low"))/pl.col("open")).mean()).item()
    worst_day_return = df.select(pl.col("ret_1d").min()).item()
    best_day_return = df.select(pl.col("ret_1d").max()).item()
    trend_strength = df.select((pl.col("ret_1d") > 0).mean()).item()
    dollar_volume_mean = df.select(pl.col("dollar_volume").mean()).item()

    amihud_mean = (
        df.select(
            pl.when(pl.col("dollar_volume") > 0).then(pl.col("ret_1d").abs()/pl.col("dollar_volume"))
            .otherwise(None)
            .mean()
        ).item()
    )

    trade_count_mean = None
    if "trade_count" in df.columns:
        trade_count_mean = df.select(pl.col("trade_count").mean()).item()
    
    close_start = df.select(pl.col("close").first()).item()
    close_end = df.select(pl.col("close").last()).item()
    cum_return = None if close_start in (None, 0) else (close_end/close_start - 1.0)

    max_drawdown_in_window = (
        df.select((pl.col("close")/pl.col("close").cum_max() - 1.0).min()).item()
    )

    def _f(x: object) -> float:
          return float(x) if x is not None else 0.0

    return {
          "realized_vol": _f(realized_vol),
          "range_mean": _f(range_mean),
          "cum_return": _f(cum_return),
          "worst_day_return": _f(worst_day_return),
          "best_day_return": _f(best_day_return),
          "trend_strength": _f(trend_strength),
          "dollar_volume_mean": _f(dollar_volume_mean),
          "amihud_mean": _f(amihud_mean),
          "trade_count_mean": _f(trade_count_mean),
          "max_drawdown_in_window": _f(max_drawdown_in_window),
      }

def compute_forward_mdd_label(
    bars_1d: pl.DataFrame,
    asof: datetime,
    horizon_days: int,
    *,
    threshold: float = -0.25,
) -> float:
    """
    Computes the forward max drawdown from (asof, asof+horizon] and returns 1.0 if MDD <= threshold else 0.0.

    This is your supervised target for calibration:
      y_h = 1{ MDD_{asof->asof+h} <= -25% }

    Tools you'll use:
    - filter future slice by ts
    - compute running peak of close
    - drawdown series = close/peak - 1
    - min drawdown over horizon

    Return is float (0.0/1.0) for convenience.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    required = {"ts", "close"}
    missing = required - set(bars_1d.columns)
    if missing:
        raise ValueError(f"bars_1d missing required columns for MDD label: {sorted(missing)}")
    
    future = bars_1d.filter(pl.col("ts")>asof).sort("ts")
    future = future.head(horizon_days)

    if future.height < horizon_days:
        return float("nan")
    
    future = future.with_columns(
        pl.col("close").cum_max().alias("peak_close")
    )

    future = future.with_columns(
        (pl.col("close") / pl.col("peak_close") - 1.0).alias("drawdown")
    )

    #Max drawdown over the horizon = min(drawdown)
    mdd = future.select(pl.col("drawdown").min()).item()
    return 1.0 if mdd <= threshold else 0.0
