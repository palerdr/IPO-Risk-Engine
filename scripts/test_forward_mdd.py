"""
Test forward MDD label computation.

1. Synthetic test with hand-computed expected values
2. Real data test with PLTR daily bars
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.labels.mdd import compute_forward_mdd


def test_synthetic():
    """Hand-computed test case to verify correctness."""
    print("=" * 60)
    print("TEST 1: Synthetic (hand-computed)")
    print("=" * 60)

    # 5 bars with known highs/lows
    # highs: [10, 12, 11,  9, 13]
    # lows:  [ 9, 11,  8,  7, 12]
    df = pl.DataFrame({
        "ts": pl.date_range(datetime(2025, 1, 1), datetime(2025, 1, 5), eager=True),
        "open":  [10.0, 11.0, 10.0, 8.0, 12.0],
        "high":  [10.0, 12.0, 11.0, 9.0, 13.0],
        "low":   [ 9.0, 11.0,  8.0, 7.0, 12.0],
        "close": [10.0, 11.5,  9.0, 8.0, 12.5],
        "volume": [100, 100, 100, 100, 100],
    })

    result = compute_forward_mdd(df, horizon=2)
    print(f"\nColumns: {result.columns}")
    print(f"Shape: {result.shape}\n")

    vals = result[f"forward_mdd_2d"].to_list()

    # Row 0: window [1,2] → highs [12,12], lows [11,8]
    #   j=1: peak=12, dd=(11-12)/12 = -0.0833
    #   j=2: peak=12, dd=(8-12)/12  = -0.3333
    #   MDD = -0.3333
    expected_0 = -1/3
    assert abs(vals[0] - expected_0) < 1e-6, f"Row 0: expected {expected_0:.4f}, got {vals[0]:.4f}"
    print(f"  Row 0: {vals[0]:.6f} (expected {expected_0:.6f}) PASS")

    # Row 1: window [2,3] → highs [11,11], lows [8,7]
    #   j=2: peak=11, dd=(8-11)/11 = -0.2727
    #   j=3: peak=11, dd=(7-11)/11 = -0.3636
    #   MDD = -0.3636
    expected_1 = -4/11
    assert abs(vals[1] - expected_1) < 1e-6, f"Row 1: expected {expected_1:.4f}, got {vals[1]:.4f}"
    print(f"  Row 1: {vals[1]:.6f} (expected {expected_1:.6f}) PASS")

    # Row 2: window [3,4] → highs [9,13], lows [7,12]
    #   j=3: peak=9,  dd=(7-9)/9   = -0.2222
    #   j=4: peak=13, dd=(12-13)/13 = -0.0769
    #   MDD = -0.2222
    expected_2 = -2/9
    assert abs(vals[2] - expected_2) < 1e-6, f"Row 2: expected {expected_2:.4f}, got {vals[2]:.4f}"
    print(f"  Row 2: {vals[2]:.6f} (expected {expected_2:.6f}) PASS")

    # Rows 3,4: insufficient forward data → None
    assert vals[3] is None, f"Row 3: expected None, got {vals[3]}"
    print(f"  Row 3: None PASS")
    assert vals[4] is None, f"Row 4: expected None, got {vals[4]}"
    print(f"  Row 4: None PASS")

    print("\nSynthetic test: ALL PASSED")


def test_real_data():
    """Test with real PLTR daily bars."""
    print("\n" + "=" * 60)
    print("TEST 2: Real data (PLTR daily)")
    print("=" * 60)

    settings = load_settings()
    symbol = "PLTR"
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=60)

    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    result = ingest_bars(symbol, start, end, tf_1d, "1d", settings=settings, force_refresh=True)
    print(f"\nIngested {result.rows} daily bars")

    df = read_parquet(raw_bars_path(symbol, "1d"))

    for horizon in [7, 20]:
        print(f"\n--- Horizon: {horizon}d ---")
        mdd_df = compute_forward_mdd(df, horizon=horizon)

        col = f"forward_mdd_{horizon}d"
        non_null = mdd_df.filter(pl.col(col).is_not_null())
        nulls = mdd_df.filter(pl.col(col).is_null())

        print(f"  Total rows:    {mdd_df.height}")
        print(f"  Non-null:      {non_null.height}")
        print(f"  Null (tail):   {nulls.height}")

        vals = non_null[col]
        print(f"  Min MDD:       {vals.min():.4f}")
        print(f"  Max MDD:       {vals.max():.4f}")
        print(f"  Mean MDD:      {vals.mean():.4f}")
        print(f"  Median MDD:    {vals.median():.4f}")

        # Sanity checks
        assert nulls.height == horizon, f"Expected {horizon} nulls, got {nulls.height}"
        print(f"  Null count == horizon: PASS")

        assert vals.max() <= 0, f"MDD should be <= 0, got max {vals.max()}"
        print(f"  All MDD <= 0: PASS")

    print("\nReal data test: ALL PASSED")


if __name__ == "__main__":
    test_synthetic()
    test_real_data()
    print("\n" + "=" * 60)
    print("All tests complete!")
