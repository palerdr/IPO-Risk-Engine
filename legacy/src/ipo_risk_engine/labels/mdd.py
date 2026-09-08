import polars as pl

def compute_forward_mdd(daily_bars: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """
    Args:
        daily_bars: DataFrame with columns [ts, open, high, low, close, volume]
                    sorted by ts ascending.
        horizon: number of forward trading days.

    Returns:
        DataFrame with columns [ts, forward_mdd_{horizon}d]
        Rows with insufficient forward data get null.
    """
    if "ts" not in daily_bars.columns or "high" not in daily_bars.columns or "low" not in daily_bars.columns:
        raise ValueError("daily_bars must have columns: ts, high, low")

    bars = daily_bars.sort("ts")
    n = bars.height
    col_name = f"forward_mdd_{horizon}d"

    highs = bars["high"].to_list()
    lows = bars["low"].to_list()

    mdd_values: list[float | None] = []
    for i in range(n):
        if n-i <= horizon:
            mdd_values.append(None)
        else:
            running_peak = None
            mdd_i = None
            for j in range(i + 1, i + horizon + 1):
                if running_peak is not None:
                    running_peak = max(running_peak, highs[j])
                else:
                    running_peak = highs[j]
                    
                if running_peak != 0:
                    drawdown_j = (lows[j] - running_peak)/running_peak
                else:
                    raise ValueError("running peak = 0, divide by 0 error")

                if mdd_i is not None:
                    mdd_i = min(mdd_i, drawdown_j)
                else:
                    mdd_i = drawdown_j

            mdd_values.append(mdd_i)

    return pl.DataFrame({
        "ts": bars["ts"],
        col_name: mdd_values,
    })

def compute_forward_max_runup(daily_bars: pl.DataFrame, horizon: int) -> pl.DataFrame:

    if "ts" not in daily_bars.columns or "high" not in daily_bars.columns or "low" not in daily_bars.columns:
        raise ValueError("daily_bars must have columns: ts, high, low")

    bars = daily_bars.sort("ts")
    n = bars.height
    col_name = f"forward_mfr_{horizon}d"

    highs = bars["high"].to_list()
    lows = bars["low"].to_list()

    mfr_values: list[float | None] = []
    for i in range(n):
        if n-i <= horizon:
            mfr_values.append(None)
        else:
            running_trough = None
            mfr_i = None
            for j in range(i + 1, i + horizon + 1):
                if running_trough is not None:
                    running_trough = min(running_trough, lows[j])
                else:
                    running_trough = lows[j]

                if running_trough != 0:
                    runup_j = (highs[j] - running_trough)/running_trough
                else:
                    raise ValueError("running trough = 0, divide by 0 error")

                if mfr_i is not None:
                    mfr_i = max(mfr_i, runup_j)
                else:
                    mfr_i = runup_j

            mfr_values.append(mfr_i)

    return pl.DataFrame({
        "ts": bars["ts"],
        col_name: mfr_values,
    })
