from datetime import datetime, timezone
import math
import statistics

import polars as pl
import pytest

from ipo_risk_engine.features.core_daily import (
    add_basic_returns,
    compute_core_features,
    compute_forward_mdd_label,
    slice_window,
)


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_slice_window_filters_and_sorts() -> None:
    df = pl.DataFrame(
        {
            "symbol": ["PLTR", "PLTR", "PLTR"],
            "ts": [_dt(2020, 9, 1), _dt(2020, 8, 31), _dt(2020, 9, 2)],
            "open": [10.0, 9.5, 10.5],
            "high": [10.2, 9.8, 10.7],
            "low": [9.8, 9.4, 10.1],
            "close": [10.1, 9.6, 10.6],
            "volume": [100, 120, 110],
        }
    )

    start = _dt(2020, 9, 1)
    end = _dt(2020, 9, 3)

    out = slice_window(df, start, end)
    assert out.height == 2
    assert out.get_column("ts").to_list() == [_dt(2020, 9, 1), _dt(2020, 9, 2)]


def test_add_basic_returns() -> None:
    df = pl.DataFrame(
        {
            "ts": [_dt(2020, 9, 1), _dt(2020, 9, 2), _dt(2020, 9, 3)],
            "open": [10.0, 11.0, 9.0],
            "close": [10.0, 11.0, 9.9],
        }
    )

    out = add_basic_returns(df)
    ret = out.get_column("ret_1d").to_list()
    logret = out.get_column("logret_1d").to_list()
    gap = out.get_column("gap_oc").to_list()
    intra = out.get_column("intraday_ret").to_list()

    assert ret[0] is None
    assert logret[0] is None
    assert gap[0] is None
    assert intra[0] == 0.0
    assert ret[1] == pytest.approx(0.1, rel=1e-9)
    assert logret[1] == pytest.approx(math.log1p(0.1), rel=1e-9)
    assert gap[1] == pytest.approx(0.1, rel=1e-9)
    assert intra[1] == 0.0


def test_compute_core_features() -> None:
    df = pl.DataFrame(
        {
            "ts": [_dt(2020, 9, 1), _dt(2020, 9, 2), _dt(2020, 9, 3)],
            "open": [10.0, 11.0, 9.0],
            "high": [10.5, 11.2, 9.5],
            "low": [9.5, 10.8, 8.5],
            "close": [10.0, 11.0, 9.9],
            "volume": [100, 200, 150],
        }
    )

    features = compute_core_features(df)

    logrets = [math.log1p(0.1), math.log1p(-0.1)]
    realized_vol = statistics.stdev(logrets)
    range_vals = [(10.5 - 9.5) / 10.0, (11.2 - 10.8) / 11.0, (9.5 - 8.5) / 9.0]
    range_mean = sum(range_vals) / len(range_vals)
    cum_return = 9.9 / 10.0 - 1.0
    worst_day_return = -0.1
    best_day_return = 0.1
    trend_strength = 1.0 / 2.0
    dollar_volume_mean = (10.0 * 100 + 11.0 * 200 + 9.9 * 150) / 3.0
    amihud_vals = [abs(0.1) / (11.0 * 200), abs(-0.1) / (9.9 * 150)]
    amihud_mean = sum(amihud_vals) / len(amihud_vals)
    max_drawdown = min([0.0, 0.0, 9.9 / 11.0 - 1.0])

    assert features["realized_vol"] == pytest.approx(realized_vol, rel=1e-9)
    assert features["range_mean"] == pytest.approx(range_mean, rel=1e-9)
    assert features["cum_return"] == pytest.approx(cum_return, rel=1e-9)
    assert features["worst_day_return"] == pytest.approx(worst_day_return, rel=1e-9)
    assert features["best_day_return"] == pytest.approx(best_day_return, rel=1e-9)
    assert features["trend_strength"] == pytest.approx(trend_strength, rel=1e-9)
    assert features["dollar_volume_mean"] == pytest.approx(dollar_volume_mean, rel=1e-9)
    assert features["amihud_mean"] == pytest.approx(amihud_mean, rel=1e-9)
    assert features["max_drawdown_in_window"] == pytest.approx(max_drawdown, rel=1e-9)


def test_compute_forward_mdd_label() -> None:
    df = pl.DataFrame(
        {
            "ts": [_dt(2020, 9, 1), _dt(2020, 9, 2), _dt(2020, 9, 3), _dt(2020, 9, 4)],
            "close": [10.0, 12.0, 9.0, 11.0],
        }
    )

    label = compute_forward_mdd_label(df, _dt(2020, 9, 1), horizon_days=3, threshold=-0.25)
    assert label == 1.0

    df2 = pl.DataFrame(
        {
            "ts": [_dt(2020, 9, 1), _dt(2020, 9, 2), _dt(2020, 9, 3)],
            "close": [10.0, 10.5, 10.2],
        }
    )
    label2 = compute_forward_mdd_label(df2, _dt(2020, 9, 1), horizon_days=2, threshold=-0.25)
    assert label2 == 0.0
