"""
Validate EDGAR universe against Alpaca daily-bar availability.

Reads ipo_universe.parquet, checks each symbol for daily bar data,
and produces two output files:
  - data/ipo_universe_all.parquet       (full EDGAR universe + alpaca_available flag)
  - data/ipo_universe_alpaca_v1.parquet (modelable subset with Alpaca data)

Usage:
    python -m scripts.validate_universe_alpaca
    python -m scripts.validate_universe_alpaca --batch-size 50
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings

UNIVERSE_PATH = Path("data/ipo_universe.parquet")
ALL_PATH = Path("data/ipo_universe_all.parquet")
V1_PATH = Path("data/ipo_universe_alpaca_v1.parquet")


def check_alpaca_batch(
    client: StockHistoricalDataClient,
    symbols: list[str],
    ipo_dates: list,
) -> dict[str, bool]:
    """Check which symbols have any daily bars on Alpaca (single-symbol calls)."""
    results: dict[str, bool] = {}

    for sym, ipo_d in zip(symbols, ipo_dates):
        start = datetime.combine(ipo_d, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=30)

        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start,
                end=end,
            ))
            bar_list = bars.data.get(sym, [])
            results[sym] = len(bar_list) > 0
        except Exception:
            results[sym] = False

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate universe against Alpaca")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Symbols per progress report")
    args = parser.parse_args()

    print("Loading EDGAR universe...")
    df = pl.read_parquet(UNIVERSE_PATH)
    print(f"  {df.height} symbols")

    settings = load_settings()
    client = StockHistoricalDataClient(
        api_key=settings.api_key, secret_key=settings.api_secret,
    )

    symbols = df["symbol"].to_list()
    ipo_dates = df["ipo_date"].to_list()
    available: dict[str, bool] = {}

    print(f"\nValidating {len(symbols)} symbols against Alpaca...")
    t0 = time.time()

    for i in range(0, len(symbols), args.batch_size):
        batch_syms = symbols[i:i + args.batch_size]
        batch_dates = ipo_dates[i:i + args.batch_size]

        batch_results = check_alpaca_batch(client, batch_syms, batch_dates)
        available.update(batch_results)

        n_done = min(i + args.batch_size, len(symbols))
        n_avail = sum(1 for v in available.values() if v)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(symbols) - n_done) / rate if rate > 0 else 0
        print(f"  [{n_done}/{len(symbols)}] available={n_avail} "
              f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Add alpaca_available column
    df_all = df.with_columns(
        pl.Series("alpaca_available", [available.get(s, False) for s in symbols])
    )
    df_v1 = df_all.filter(pl.col("alpaca_available"))

    # Save both
    ALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.write_parquet(ALL_PATH)
    df_v1.write_parquet(V1_PATH)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total EDGAR universe:  {df_all.height}")
    print(f"  Alpaca available:      {df_v1.height} ({df_v1.height/df_all.height:.1%})")
    print(f"  Not available:         {df_all.height - df_v1.height}")

    # Coverage by era
    print(f"\n  Coverage by era:")
    for era, yr_start, yr_end in [("2010-2016", 2010, 2016), ("2017-2021", 2017, 2021), ("2022-2025", 2022, 2025)]:
        era_all = df_all.filter(
            (pl.col("ipo_date").dt.year() >= yr_start) &
            (pl.col("ipo_date").dt.year() <= yr_end)
        )
        era_v1 = era_all.filter(pl.col("alpaca_available"))
        pct = era_v1.height / era_all.height * 100 if era_all.height > 0 else 0
        print(f"    {era}: {era_v1.height}/{era_all.height} ({pct:.1f}%)")

    print(f"\n  Wrote: {ALL_PATH} ({df_all.height} rows)")
    print(f"  Wrote: {V1_PATH} ({df_v1.height} rows)")


if __name__ == "__main__":
    main()
