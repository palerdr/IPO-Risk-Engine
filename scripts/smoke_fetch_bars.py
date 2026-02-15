from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.alpaca_client import AlpacaData


def main():
    symbol = "PLTR"

    # Using UTC timestamps because APIs commonly accept UTC.
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=7)
    settings = load_settings()
    alpaca = AlpacaData.from_settings(settings)

    bars = alpaca.get_daily_bars(symbol=symbol, start=start, end=end)

    pdf = bars.df.reset_index()

    # Convert to Polars
    df = pl.from_pandas(pdf)

    out_dir = Path("data/raw") / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_bars.parquet"

    df.write_parquet(out_path)

    print(f"Wrote {df.shape[0]} rows to {out_path}")


if __name__ == "__main__":
    main()
