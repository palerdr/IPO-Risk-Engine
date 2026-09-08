"""
Test schema validation against real Alpaca data.
"""
from datetime import datetime, timedelta
import polars as pl
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.alpaca_client import AlpacaData
from ipo_risk_engine.data.schemas import validate_bars


def bars_to_dataframe(bars, symbol: str) -> pl.DataFrame:
    """Convert Alpaca BarSet to Polars DataFrame."""
    if symbol not in bars.data:
        raise ValueError(f"No data for {symbol}")

    bar_list = bars.data[symbol]

    # Extract fields from each bar
    records = []
    for bar in bar_list:
        records.append({
            "ts": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "vwap": float(bar.vwap) if bar.vwap else None,
            "trade_count": int(bar.trade_count) if bar.trade_count else None,
        })

    return pl.DataFrame(records)


def main():
    settings = load_settings()
    client = AlpacaData.from_settings(settings)

    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=3)
    symbol = "PLTR"

    print(f"Testing schema validation for {symbol}")
    print("=" * 50)

    # Test 5-minute bars
    print("\n[1] 5-minute bars:")
    tf_5m = TimeFrame(5, TimeFrameUnit.Minute)
    bars_5m = client.get_bars(symbol, tf_5m, start, end)
    df_5m = bars_to_dataframe(bars_5m, symbol)

    print(f"    DataFrame schema: {df_5m.schema}")
    try:
        validate_bars(df_5m, "5m")
        print("    ✓ Validation passed!")
    except (ValueError, TypeError) as e:
        print(f"    ✗ Validation failed: {e}")

    # Test daily bars
    print("\n[2] Daily bars:")
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    bars_1d = client.get_bars(symbol, tf_1d, start, end)
    df_1d = bars_to_dataframe(bars_1d, symbol)

    print(f"    DataFrame schema: {df_1d.schema}")
    try:
        validate_bars(df_1d, "1d")
        print("    ✓ Validation passed!")
    except (ValueError, TypeError) as e:
        print(f"    ✗ Validation failed: {e}")

    print("\n" + "=" * 50)
    print("Schema validation test complete!")


if __name__ == "__main__":
    main()
