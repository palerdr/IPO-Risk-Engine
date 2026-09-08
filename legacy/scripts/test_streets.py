"""
Test street windows with multi-granularity mapping.
"""
from datetime import date

from ipo_risk_engine.features.streets import (
    compute_street_windows,
    get_timeframe_key,
    STREET_TIMEFRAME,
)


def main():
    print("Street -> Timeframe Mapping")
    print("=" * 40)
    for street, tf_key in STREET_TIMEFRAME.items():
        print(f"  {street:8} -> {tf_key}")

    print("\n" + "=" * 40)
    print("Example: IPO listing on 2026-01-15")
    print("=" * 40)

    # Simulate trading days for an IPO (skip weekends)
    # Jan 15 = Wed, so trading days are: 15,16,17, 21,22,23,24, 27,28,29,30,31, Feb 3,4,5,6,7...
    trading_days = [
        date(2026, 1, 15), date(2026, 1, 16), date(2026, 1, 17),  # Wed-Fri
        date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23), date(2026, 1, 24),  # Mon-Fri
        date(2026, 1, 27), date(2026, 1, 28), date(2026, 1, 29), date(2026, 1, 30), date(2026, 1, 31),
        date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5), date(2026, 2, 6), date(2026, 2, 7),
        date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12), date(2026, 2, 13),
    ]

    listing_day = trading_days[0]

    windows = compute_street_windows(
        listing_day=listing_day,
        trading_days=trading_days,
        flop_days=1,   # Day 0
        turn_days=5,   # Days 1-5
        river_days=15, # Days 6-20
    )

    print(f"\nGenerated {len(windows)} street windows:\n")

    for w in windows:
        tf_key = get_timeframe_key(w.street)
        print(f"  {w.street:8}")
        print(f"    Window:    {w.start.date()} to {w.end.date()} (exclusive)")
        print(f"    Bars file: bars_{tf_key}.parquet")
        print()

    print("=" * 40)
    print("Test complete!")


if __name__ == "__main__":
    main()
