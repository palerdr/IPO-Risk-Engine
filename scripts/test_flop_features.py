"""
Test FLOP feature computation on real 5-minute bar data.
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.features.street_features import compute_flop_features


def main():
    settings = load_settings()

    # Use a recent trading day - need full day range to capture market hours
    symbol = "PLTR"
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=5)  # Get several days to ensure we have full market hours

    print(f"Testing FLOP features for {symbol}")
    print(f"Date range: {start.date()} to {end.date()}")
    print("=" * 60)

    # Ingest 5-minute bars
    tf_5m = TimeFrame(5, TimeFrameUnit.Minute)
    result = ingest_bars(symbol, start, end, tf_5m, "5m", settings=settings, force_refresh=True)
    print(f"\nIngested {result.rows} 5-minute bars")

    # Load the bars
    df = read_parquet(raw_bars_path(symbol, "5m"))
    print(f"Loaded DataFrame with {df.height} rows")
    print(f"Time range: {df['ts'].min()} to {df['ts'].max()}")

    # For FLOP test, find a day that has market hours data (14:30-21:00 UTC)
    import polars as pl
    from datetime import time
    MARKET_OPEN_UTC = time(14, 30)
    MARKET_CLOSE_UTC = time(21, 0)

    # Filter to market hours only
    df_market = df.filter(
        (pl.col("ts").dt.time() >= MARKET_OPEN_UTC) &
        (pl.col("ts").dt.time() < MARKET_CLOSE_UTC)
    )
    print(f"\nBars within market hours: {df_market.height}")

    if df_market.height == 0:
        print("ERROR: No bars within market hours in dataset!")
        return

    unique_dates = df_market.select(pl.col("ts").dt.date().unique()).sort("ts")["ts"].to_list()
    print(f"Dates with market hours data: {unique_dates}")

    # Pick first available date with market hours
    test_date = unique_dates[0]
    df_one_day = df.filter(pl.col("ts").dt.date() == test_date)
    print(f"Testing with {test_date}: {df_one_day.height} bars")
    print(f"Time range: {df_one_day['ts'].min()} to {df_one_day['ts'].max()}")

    # Compute features
    print("\nComputing FLOP features...")
    try:
        features = compute_flop_features(df_one_day)

        print("\nFLOP Features:")
        print("-" * 40)
        for name, value in features.items():
            if isinstance(value, float):
                print(f"  {name:25}: {value:>12.4f}")
            else:
                print(f"  {name:25}: {value}")

        # Sanity checks
        print("\nSanity Checks:")
        print("-" * 40)

        vol_pct = features['volume_first_hour_pct']
        if 0 < vol_pct < 1:
            print(f"  volume_first_hour_pct in (0,1): PASS ({vol_pct:.2%})")
        else:
            print(f"  volume_first_hour_pct in (0,1): FAIL ({vol_pct})")

        mdd = features['intraday_mdd']
        if mdd <= 0:
            print(f"  intraday_mdd <= 0: PASS ({mdd:.4f})")
        else:
            print(f"  intraday_mdd <= 0: FAIL ({mdd})")

        ttm = features['time_to_high_minutes']
        if 0 <= ttm <= 390:  # Market is 6.5 hours = 390 minutes
            print(f"  time_to_high in [0, 390]: PASS ({ttm:.1f} min)")
        else:
            print(f"  time_to_high in [0, 390]: FAIL ({ttm})")

    except Exception as e:
        print(f"Error computing features: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
