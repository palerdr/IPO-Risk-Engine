from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from alpaca.data.timeframe import TimeFrame

from ipo_risk_engine.config.settings import AlpacaSettings
from ipo_risk_engine.data.alpaca_client import AlpacaData
from ipo_risk_engine.data.store import raw_bars_path, write_parquet, read_parquet
from ipo_risk_engine.data.schemas import validate_bars

@dataclass(frozen=True)
class IngestResult:
    symbol: str
    path: Path
    rows: int
    from_cache: bool


def normalize_bars(df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """
    Canonical bars schema:

      symbol: str
      ts: datetime (UTC)
      open/high/low/close: float
      volume: int
      vwap: float (optional but present here)
      trade_count: int (optional but present here)
    """
    
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected column 'timestamp' in bars df, got: {df.columns}")
    df = df.rename({"timestamp": "ts"})
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
    else:
        df = df.with_columns(pl.col("symbol").cast(pl.Utf8).str.to_uppercase())
    df = df.with_columns(
        [
            pl.col("ts").cast(pl.Datetime(time_zone="UTC")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
        ]
    )
    # Only cast optional fields if present
    if "vwap" in df.columns:
        df = df.with_columns(pl.col("vwap").cast(pl.Float64))
    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").cast(pl.Int64))
    if "trade_count" in df.columns:
        df = df.with_columns(pl.col("trade_count").cast(pl.Int64))
    cols = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    if "vwap" in df.columns:
        cols.append("vwap")
    if "trade_count" in df.columns:
        cols.append("trade_count")
    df = df.select(cols)
    df = df.unique(subset = ["symbol", 'ts'], keep='last').sort(["symbol", 'ts'])
    return df

def ingest_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: TimeFrame,
    timeframe_key: str,  # "5m", "1h", "1d"
    *,
    settings: AlpacaSettings,
    force_refresh: bool = False,
) -> IngestResult:
    symbol = symbol.upper()
    path = raw_bars_path(symbol, timeframe_key)
    if path.exists() and not force_refresh:
        df = read_parquet(path)
        return IngestResult(symbol=symbol, path=path, rows=df.height, from_cache=True)
    
    alpaca = AlpacaData.from_settings(settings)
    bars = alpaca.get_bars(symbol, timeframe, start, end)

    pdf = bars.df.reset_index()
    if pdf.empty or "timestamp" not in pdf.columns:
        raise ValueError(f"No bar data returned by Alpaca for {symbol}")
    df = pl.from_pandas(pdf)
    df = normalize_bars(df, symbol)

    validate_bars(df, timeframe_key)
    write_parquet(df, path)
    return IngestResult(symbol=symbol, path=path, rows=df.height, from_cache=False)
    


# def ingest_daily_bars(
#     symbol: str,
#     start: datetime,
#     end: datetime,
#     *,
#     settings: AlpacaSettings,
#     force_refresh: bool = False,
# ) -> IngestResult:
#     """
#     Fetches daily bars from Alpaca, normalizes schema, and caches to Parquet.
#     """
#     symbol = symbol.upper()
#     path = raw_bars_path(symbol, )

#     if path.exists() and not force_refresh:
#         df = read_parquet(path)
#         return IngestResult(symbol=symbol, path=path, rows=df.height, from_cache=True)

#     alpaca = AlpacaData.from_settings(settings)
#     bars = alpaca.get_daily_bars(symbol=symbol, start=start, end=end)

#     # We'll convert to pandas first, then to polars (boundary layer).
#     pdf = bars.df.reset_index()
#     df = pl.from_pandas(pdf)

#     df = normalize_daily_bars(df, symbol=symbol)

#     write_parquet(df, path)
#     return IngestResult(symbol=symbol, path=path, rows=df.height, from_cache=False)

