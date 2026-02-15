from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from ipo_risk_engine.data.store import read_parquet, write_parquet
from ipo_risk_engine.features.calendar import trading_days_from_bars_1d
from ipo_risk_engine.features.core_daily import compute_core_features, compute_forward_mdd_label, slice_window
from ipo_risk_engine.features.streets import Street, StreetWindow, compute_street_windows


SNAPSHOT_DIR = Path("data") / "features"
SNAPSHOT_PATH = SNAPSHOT_DIR / "snapshots_1d.parquet"


def _asof_from_window(window: StreetWindow) -> datetime:
    # The as-of timestamp is the last day in the window.
    return window.end - timedelta(days=1)


def build_snapshot_row(
    symbol: str,
    street: Street,
    asof: datetime,
    bars_1d: pl.DataFrame,
    *,
    street_window: StreetWindow | None = None,
    horizons: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, object]:
    if street_window is not None:
        window_bars = slice_window(bars_1d, street_window.start, street_window.end)
    else:
        window_bars = bars_1d.filter(pl.col("ts") <= asof).sort(["symbol", "ts"])

    features = compute_core_features(window_bars)
    labels: dict[str, float] = {}
    for h in horizons:
        label = compute_forward_mdd_label(bars_1d, asof, h, threshold=-0.25)
        labels[f"y_{h}"] = float(label)

    row: dict[str, object] = {
        "symbol": symbol,
        "street": street,
        "asof": asof,
    }
    row.update(features)
    row.update(labels)
    return row


def build_snapshot_table(
    symbols: Sequence[str],
    *,
    data_dir: Path | None = None,
    flop_days: int = 1,
    turn_days: int = 5,
    river_days: int = 15,
    horizons: Iterable[int] = (1, 3, 5, 10),
    write_out: bool = True,
    output_path: Path = SNAPSHOT_PATH,
) -> pl.DataFrame:
    base_dir = data_dir or Path("data")
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        symbol = symbol.upper()
        path = base_dir / "raw" / symbol / "bars_1d.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached bars for {symbol}: {path}")
        bars = read_parquet(path)
        days = trading_days_from_bars_1d(bars)
        windows = compute_street_windows(
            listing_day=days[0],
            trading_days=days,
            flop_days=flop_days,
            turn_days=turn_days,
            river_days=river_days,
        )

        for window in windows:
            window_bars = slice_window(bars, window.start, window.end)
            if window_bars.is_empty():
                continue
            asof = window_bars.select(pl.col("ts").max()).item()
            row = build_snapshot_row(
                symbol,
                window.street,
                asof,
                bars,
                street_window=window,
                horizons=horizons,
            )
            rows.append(row)

    table = pl.DataFrame(rows) if rows else pl.DataFrame()
    if write_out:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(table, output_path)
    return table
