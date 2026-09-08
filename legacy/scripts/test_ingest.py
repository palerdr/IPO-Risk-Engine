"""
Test the full ingest pipeline with multi-timeframe support.
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_bars


def main():
    settings = load_settings()

    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=5)
    symbol = "PLTR"

    print(f"Testing ingest for {symbol}")
    print(f"Date range: {start.date()} to {end.date()}")
    print("=" * 60)

    # Test 5-minute ingestion
    print("\n[1] Ingesting 5-minute bars...")
    tf_5m = TimeFrame(5, TimeFrameUnit.Minute)
    result_5m = ingest_bars(symbol, start, end, tf_5m, "5m", settings=settings, force_refresh=True)
    print(f"    Path: {result_5m.path}")
    print(f"    Rows: {result_5m.rows}")
    print(f"    From cache: {result_5m.from_cache}")

    # Test hourly ingestion
    print("\n[2] Ingesting hourly bars...")
    tf_1h = TimeFrame(1, TimeFrameUnit.Hour)
    result_1h = ingest_bars(symbol, start, end, tf_1h, "1h", settings=settings, force_refresh=True)
    print(f"    Path: {result_1h.path}")
    print(f"    Rows: {result_1h.rows}")
    print(f"    From cache: {result_1h.from_cache}")

    # Test daily ingestion
    print("\n[3] Ingesting daily bars...")
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    result_1d = ingest_bars(symbol, start, end, tf_1d, "1d", settings=settings, force_refresh=True)
    print(f"    Path: {result_1d.path}")
    print(f"    Rows: {result_1d.rows}")
    print(f"    From cache: {result_1d.from_cache}")

    # Test cache hit
    print("\n[4] Testing cache (should be from_cache=True)...")
    result_cached = ingest_bars(symbol, start, end, tf_1d, "1d", settings=settings, force_refresh=False)
    print(f"    From cache: {result_cached.from_cache}")

    print("\n" + "=" * 60)
    print("Ingest test complete!")


if __name__ == "__main__":
    main()
