from __future__ import annotations

from datetime import timezone
import polars as pl

from ipo_risk_engine.features.core_daily import slice_window, add_basic_returns
from ipo_risk_engine.features.calendar import trading_days_from_bars_1d
from ipo_risk_engine.features.streets import compute_street_windows


def main():
    bars = pl.read_parquet("data/raw/PLTR/bars_1d.parquet")
    days = trading_days_from_bars_1d(bars)
    listing_day = days[0]

    windows = compute_street_windows(
        listing_day=listing_day,
        trading_days=days,
        flop_days=1,
        turn_days=5,
        river_days=15,
    )

    flop = windows[0]
    w = slice_window(bars, flop.start, flop.end)
    w2 = add_basic_returns(w)

    print("FLOP window rows:", w2.height)
    print(w2.select(["symbol", "ts", "open", "close", "ret_1d", "gap_oc", "intraday_ret"]).head(5))


if __name__ == "__main__":
    main()
