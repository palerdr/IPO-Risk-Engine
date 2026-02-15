from __future__ import annotations

from datetime import date
from typing import Sequence

import polars as pl


def trading_days_from_bars_1d(bars_1d: pl.DataFrame) -> list[date]:
    """
    Extracts ordered trading dates from canonical daily bars.

    bars_1d must have:
      - ts: datetime (UTC)
    """
    if "ts" not in bars_1d.columns:
        raise ValueError("bars_1d must contain 'ts' column")

    # Convert timestamps to dates and take unique sorted values
    days = (
        bars_1d.select(pl.col("ts").dt.date().alias("d"))
        .unique()
        .sort("d")
        .get_column("d")
        .to_list()
    )
    return days
