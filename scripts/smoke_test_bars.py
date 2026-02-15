"""
Smoke test for multi-timeframe bar ingestion.
Verifies that get_bars() works with 5-minute, hourly, and daily timeframes.
"""
from datetime import datetime, timedelta
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.alpaca_client import AlpacaData


def main():
    settings = load_settings()
    print(f"Data feed from settings: {settings.data_feed}")

    client = AlpacaData.from_settings(settings)

    # Use a date range from last week (ensures data exists and is >15 min old for SIP)
    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=5)

    symbol = "PLTR"

    print(f"\nTesting {symbol} from {start.date()} to {end.date()}")
    print("=" * 60)

    # Test 1: 5-minute bars (FLOP granularity)
    print("\n[1] 5-minute bars:")
    tf_5m = TimeFrame(5, TimeFrameUnit.Minute)
    bars_5m = client.get_bars(symbol, tf_5m, start, end)

    if symbol in bars_5m.data:
        bar_list = bars_5m.data[symbol]
        print(f"    Count: {len(bar_list)} bars")
        print(f"    First: {bar_list[0].timestamp}")
        print(f"    Last:  {bar_list[-1].timestamp}")
    else:
        print("    No data returned!")

    # Test 2: Hourly bars (TURN granularity)
    print("\n[2] Hourly bars:")
    tf_1h = TimeFrame(1, TimeFrameUnit.Hour)
    bars_1h = client.get_bars(symbol, tf_1h, start, end)

    if symbol in bars_1h.data:
        bar_list = bars_1h.data[symbol]
        print(f"    Count: {len(bar_list)} bars")
        print(f"    First: {bar_list[0].timestamp}")
        print(f"    Last:  {bar_list[-1].timestamp}")
    else:
        print("    No data returned!")

    # Test 3: Daily bars (RIVER granularity)
    print("\n[3] Daily bars:")
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    bars_1d = client.get_bars(symbol, tf_1d, start, end)

    if symbol in bars_1d.data:
        bar_list = bars_1d.data[symbol]
        print(f"    Count: {len(bar_list)} bars")
        print(f"    First: {bar_list[0].timestamp}")
        print(f"    Last:  {bar_list[-1].timestamp}")
    else:
        print("    No data returned!")

    print("\n" + "=" * 60)
    print("Smoke test complete!")


if __name__ == "__main__":
    main()
