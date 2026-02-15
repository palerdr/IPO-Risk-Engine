"""
Test RIVER feature computation on real daily bar data.
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.features.street_features import compute_river_features


def main():
    settings = load_settings()

    symbol = "PLTR"
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=30)  # Get enough days for RIVER (Days 6-20 = 15 days)

    print(f"Testing RIVER features for {symbol}")
    print(f"Date range: {start.date()} to {end.date()}")
    print("=" * 60)

    # Ingest daily bars
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    result = ingest_bars(symbol, start, end, tf_1d, "1d", settings=settings, force_refresh=True)
    print(f"\nIngested {result.rows} daily bars")

    # Load the bars
    df = read_parquet(raw_bars_path(symbol, "1d"))
    print(f"Loaded DataFrame with {df.height} rows")

    # Get unique dates and pick a 15-day window (simulating Days 6-20)
    unique_dates = df.select(pl.col("ts").dt.date().unique()).sort("ts")["ts"].to_list()
    print(f"\nAvailable dates: {unique_dates}")

    # Pick 15 consecutive days (skip first few which may be partial)
    if len(unique_dates) >= 17:
        test_dates = unique_dates[1:16]  # Days 6-20 simulation (15 days)
    else:
        test_dates = unique_dates[1:-1] if len(unique_dates) > 2 else unique_dates

    print(f"Using dates for RIVER: {test_dates}")

    df_river = df.filter(pl.col("ts").dt.date().is_in(test_dates))
    print(f"Filtered to {df_river.height} bars")

    # Compute features
    print("\nComputing RIVER features...")
    try:
        features = compute_river_features(df_river)

        print("\nRIVER Features:")
        print("-" * 40)
        for name, value in features.items():
            if isinstance(value, float):
                print(f"  {name:25}: {value:>12.6f}")
            else:
                print(f"  {name:25}: {value}")

        # Sanity checks
        print("\nSanity Checks:")
        print("-" * 40)

        vol = features['realized_vol']
        if vol is None or vol >= 0:
            vol_str = f"{vol:.6f}" if vol else "None"
            print(f"  realized_vol >= 0: PASS ({vol_str})")
        else:
            print(f"  realized_vol >= 0: FAIL ({vol})")

        mdd = features['max_drawdown_river']
        if mdd <= 0:
            print(f"  max_drawdown_river <= 0: PASS ({mdd:.4f})")
        else:
            print(f"  max_drawdown_river <= 0: FAIL ({mdd})")

        amihud = features['amihud_illiquidity']
        if amihud is None or amihud >= 0:
            amihud_str = f"{amihud:.10f}" if amihud else "None"
            print(f"  amihud_illiquidity >= 0: PASS ({amihud_str})")
        else:
            print(f"  amihud_illiquidity >= 0: FAIL ({amihud})")

        worst = features['worst_day_return']
        best = features['best_day_return']
        if worst is None or best is None or worst <= best:
            worst_str = f"{worst:.4f}" if worst else "None"
            best_str = f"{best:.4f}" if best else "None"
            print(f"  worst_day <= best_day: PASS (worst={worst_str}, best={best_str})")
        else:
            print(f"  worst_day <= best_day: FAIL (worst={worst}, best={best})")

        dollar_vol = features['dollar_volume_mean']
        if dollar_vol is None or dollar_vol > 0:
            dollar_str = f"{dollar_vol:,.0f}" if dollar_vol else "None"
            print(f"  dollar_volume_mean > 0: PASS ({dollar_str})")
        else:
            print(f"  dollar_volume_mean > 0: FAIL ({dollar_vol})")

    except Exception as e:
        print(f"Error computing features: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
