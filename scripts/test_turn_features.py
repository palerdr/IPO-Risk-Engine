"""
Test TURN feature computation on real hourly bar data.
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.features.street_features import compute_turn_features


def main():
    settings = load_settings()

    symbol = "PLTR"
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=10)  # Get enough days for TURN

    print(f"Testing TURN features for {symbol}")
    print(f"Date range: {start.date()} to {end.date()}")
    print("=" * 60)

    # Ingest hourly bars
    tf_1h = TimeFrame(1, TimeFrameUnit.Hour)
    result = ingest_bars(symbol, start, end, tf_1h, "1h", settings=settings, force_refresh=True)
    print(f"\nIngested {result.rows} hourly bars")

    # Load the bars
    df = read_parquet(raw_bars_path(symbol, "1h"))
    print(f"Loaded DataFrame with {df.height} rows")

    # Get unique dates and pick a 5-day window (simulating Days 1-5)
    unique_dates = df.select(pl.col("ts").dt.date().unique()).sort("ts")["ts"].to_list()
    print(f"\nAvailable dates: {unique_dates}")

    # Pick 5 consecutive days (skip first and last which may be partial)
    if len(unique_dates) >= 7:
        test_dates = unique_dates[1:6]  # Days 1-5 simulation
    else:
        test_dates = unique_dates[1:-1] if len(unique_dates) > 2 else unique_dates

    print(f"Using dates for TURN: {test_dates}")

    df_turn = df.filter(pl.col("ts").dt.date().is_in(test_dates))
    print(f"Filtered to {df_turn.height} bars")

    # Compute features
    print("\nComputing TURN features...")
    try:
        features = compute_turn_features(df_turn)

        print("\nTURN Features:")
        print("-" * 40)
        for name, value in features.items():
            if isinstance(value, float):
                print(f"  {name:25}: {value:>12.4f}")
            else:
                print(f"  {name:25}: {value}")

        # Sanity checks
        print("\nSanity Checks:")
        print("-" * 40)

        vol = features['hourly_realized_vol']
        if vol >= 0:
            print(f"  hourly_realized_vol >= 0: PASS ({vol:.4f})")
        else:
            print(f"  hourly_realized_vol >= 0: FAIL ({vol})")

        mdd = features['max_drawdown_turn']
        if mdd <= 0:
            print(f"  max_drawdown_turn <= 0: PASS ({mdd:.4f})")
        else:
            print(f"  max_drawdown_turn <= 0: FAIL ({mdd})")

        gfr = features['gap_fill_ratio']
        if 0 <= gfr <= 1:
            print(f"  gap_fill_ratio in [0,1]: PASS ({gfr:.2%})")
        else:
            print(f"  gap_fill_ratio in [0,1]: FAIL ({gfr})")

    except Exception as e:
        print(f"Error computing features: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
