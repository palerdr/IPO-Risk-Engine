from __future__ import annotations

import polars as pl

from ipo_risk_engine.features.calendar import trading_days_from_bars_1d
from ipo_risk_engine.features.streets import compute_street_windows


def main():
    bars = pl.read_parquet("data/raw/PLTR/bars_1d.parquet")
    days = trading_days_from_bars_1d(bars)

    listing_day = days[0]  # For MVP, treat first day in our dataset as "listing day"
    windows = compute_street_windows(
        listing_day=listing_day,
        trading_days=days,
        flop_days=1,
        turn_days=5,
        river_days=15,  # days 6–20 inclusive is 15 trading days after the first 6 days; this is the MVP approx
    )

    for w in windows:
        print(w)


if __name__ == "__main__":
    main()
