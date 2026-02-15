"""
Test forward MDD and MFR (max forward runup) label computation.

1. Synthetic tests with hand-computed expected values
2. Real data tests with PLTR daily bars
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.labels.mdd import compute_forward_mdd, compute_forward_max_runup


def test_mdd_synthetic():
    """Hand-computed MDD test case."""
    print("=" * 60)
    print("TEST 1: MDD Synthetic (hand-computed)")
    print("=" * 60)

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
    vals = result["forward_mdd_2d"].to_list()

    # Row 0: window [1,2] → peak tracks 12, lows [11,8] → MDD = (8-12)/12 = -1/3
    expected_0 = -1/3
    assert abs(vals[0] - expected_0) < 1e-6, f"Row 0: expected {expected_0:.4f}, got {vals[0]:.4f}"
    print(f"  Row 0: {vals[0]:.6f} (expected {expected_0:.6f}) PASS")

    # Row 1: window [2,3] → peak=11, lows [8,7] → MDD = (7-11)/11 = -4/11
    expected_1 = -4/11
    assert abs(vals[1] - expected_1) < 1e-6
    print(f"  Row 1: {vals[1]:.6f} (expected {expected_1:.6f}) PASS")

    # Row 2: window [3,4] → peak goes 9→13, lows [7,12] → MDD = (7-9)/9 = -2/9
    expected_2 = -2/9
    assert abs(vals[2] - expected_2) < 1e-6
    print(f"  Row 2: {vals[2]:.6f} (expected {expected_2:.6f}) PASS")

    assert vals[3] is None and vals[4] is None
    print(f"  Rows 3,4: None PASS")

    print("\nMDD Synthetic: ALL PASSED")


def test_mfr_synthetic():
    """Hand-computed MFR test case."""
    print("\n" + "=" * 60)
    print("TEST 2: MFR Synthetic (hand-computed)")
    print("=" * 60)

    # Same data:
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

    result = compute_forward_max_runup(df, horizon=2)
    vals = result["forward_mfr_2d"].to_list()

    # Row 0: window [1,2], trough tracks min(lows), highs [12,11]
    #   j=1: trough=11, runup=(12-11)/11 = 1/11 = 0.0909
    #   j=2: trough=min(11,8)=8, runup=(11-8)/8 = 3/8 = 0.375
    #   MFR = max(0.0909, 0.375) = 0.375
    expected_0 = 3/8
    assert abs(vals[0] - expected_0) < 1e-6, f"Row 0: expected {expected_0:.4f}, got {vals[0]:.4f}"
    print(f"  Row 0: {vals[0]:.6f} (expected {expected_0:.6f}) PASS")

    # Row 1: window [2,3], highs [11,9], lows [8,7]
    #   j=2: trough=8, runup=(11-8)/8 = 0.375
    #   j=3: trough=min(8,7)=7, runup=(9-7)/7 = 2/7 = 0.2857
    #   MFR = max(0.375, 0.2857) = 0.375
    expected_1 = 3/8
    assert abs(vals[1] - expected_1) < 1e-6, f"Row 1: expected {expected_1:.4f}, got {vals[1]:.4f}"
    print(f"  Row 1: {vals[1]:.6f} (expected {expected_1:.6f}) PASS")

    # Row 2: window [3,4], highs [9,13], lows [7,12]
    #   j=3: trough=7, runup=(9-7)/7 = 2/7 = 0.2857
    #   j=4: trough=min(7,12)=7, runup=(13-7)/7 = 6/7 = 0.8571
    #   MFR = max(0.2857, 0.8571) = 0.8571
    expected_2 = 6/7
    assert abs(vals[2] - expected_2) < 1e-6, f"Row 2: expected {expected_2:.4f}, got {vals[2]:.4f}"
    print(f"  Row 2: {vals[2]:.6f} (expected {expected_2:.6f}) PASS")

    assert vals[3] is None and vals[4] is None
    print(f"  Rows 3,4: None PASS")

    print("\nMFR Synthetic: ALL PASSED")


def test_real_data():
    """Test both MDD and MFR with real PLTR daily bars."""
    print("\n" + "=" * 60)
    print("TEST 3: Real data (PLTR daily)")
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
        mfr_df = compute_forward_max_runup(df, horizon=horizon)

        mdd_col = f"forward_mdd_{horizon}d"
        mfr_col = f"forward_mfr_{horizon}d"

        mdd_vals = mdd_df.filter(pl.col(mdd_col).is_not_null())[mdd_col]
        mfr_vals = mfr_df.filter(pl.col(mfr_col).is_not_null())[mfr_col]

        print(f"  MDD  - min: {mdd_vals.min():.4f}, max: {mdd_vals.max():.4f}, mean: {mdd_vals.mean():.4f}")
        print(f"  MFR  - min: {mfr_vals.min():.4f}, max: {mfr_vals.max():.4f}, mean: {mfr_vals.mean():.4f}")

        # Sanity checks
        assert mdd_vals.max() <= 0, f"MDD should be <= 0"
        print(f"  MDD <= 0: PASS")

        assert mfr_vals.min() >= 0, f"MFR should be >= 0"
        print(f"  MFR >= 0: PASS")

    print("\nReal data test: ALL PASSED")


if __name__ == "__main__":
    test_mdd_synthetic()
    test_mfr_synthetic()
    test_real_data()
    print("\n" + "=" * 60)
    print("All tests complete!")
