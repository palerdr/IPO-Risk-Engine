from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Sequence


Street = Literal["PREFLOP", "FLOP", "TURN", "RIVER"]
STREET_TIMEFRAME: dict[Street, str] = {
    "PREFLOP": "none",
    "FLOP":    "1d",
    "TURN":    "1d",
    "RIVER":   "1d",
}


@dataclass(frozen=True)
class StreetWindow:
    street: Street
    start: datetime
    end: datetime


def get_timeframe_key(street: Street) -> str:
    return STREET_TIMEFRAME[street]


def compute_street_windows(
    listing_day: date,
    trading_days: Sequence[date],
    *,
    flop_days: int = 6,
    turn_days: int = 15,
    river_days: int = 40,
) -> list[StreetWindow]:
    """Compute daily street windows as half-open UTC ranges [start, end)."""
    if not trading_days:
        raise ValueError("trading_days cannot be empty")

    if trading_days[0] != listing_day:
        raise ValueError(
            f"trading_days must start at listing_day. Got trading_days[0]={trading_days[0]} listing_day={listing_day}"
        )

    if flop_days < 1 or turn_days < 0 or river_days < 0:
        raise ValueError("Invalid street lengths")

    total_needed = flop_days + turn_days + river_days
    if len(trading_days) < total_needed:
        if len(trading_days) < flop_days:
            raise ValueError("Not enough trading days to form FLOP window")
        total_needed = len(trading_days)

    def day_start_utc(d: date) -> datetime:
        return datetime.combine(d, time(0, 0), tzinfo=timezone.utc)

    idx0 = 0
    idx1 = min(len(trading_days), flop_days)
    idx2 = min(len(trading_days), idx1 + turn_days)
    idx3 = min(len(trading_days), idx2 + river_days)

    windows: list[StreetWindow] = []

    windows.append(
        StreetWindow(
            street="FLOP",
            start=day_start_utc(trading_days[idx0]),
            end=day_start_utc(trading_days[idx1 - 1]) + timedelta(days=1),
        )
    )

    if idx2 > idx1:
        windows.append(
            StreetWindow(
                street="TURN",
                start=day_start_utc(trading_days[idx1]),
                end=day_start_utc(trading_days[idx2 - 1]) + timedelta(days=1),
            )
        )

    if idx3 > idx2:
        windows.append(
            StreetWindow(
                street="RIVER",
                start=day_start_utc(trading_days[idx2]),
                end=day_start_utc(trading_days[idx3 - 1]) + timedelta(days=1),
            )
        )

    return windows
