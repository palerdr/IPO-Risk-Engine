"""Construct day-20 observations without reading future bars into features."""
from datetime import date
import numpy as np

OBSERVATION_SESSIONS = 20
HORIZON_SESSIONS = 20
EVENT_THRESHOLD = 0.20
FEATURES = (
    "return_19_sessions", "return_5_sessions", "daily_volatility", "observed_drawdown",
    "worst_daily_return", "volume_change", "market_return", "market_volatility",
)


def validate_bars(bars: list[dict]) -> None:
    dates = [date.fromisoformat(row["date"]) for row in bars]
    if dates != sorted(set(dates)):
        raise ValueError("Price dates must be unique and increasing")
    for row in bars:
        close, volume = row.get("adjusted_close"), row.get("volume")
        if close is None or not np.isfinite(close) or close <= 0:
            raise ValueError(f"Invalid adjusted close on {row['date']}")
        if volume is None or not np.isfinite(volume) or volume <= 0:
            raise ValueError(f"Invalid volume on {row['date']}")


def drawdown(prices: np.ndarray) -> float:
    return float(np.max(1 - prices / np.maximum.accumulate(prices)))


def make_features(observed: list[dict], benchmark: list[dict]) -> dict[str, float]:
    if len(observed) != OBSERVATION_SESSIONS:
        raise ValueError("Supply exactly 20 observed sessions")
    validate_bars(observed)
    benchmark = [row for row in benchmark if observed[0]["date"] <= row["date"] <= observed[-1]["date"]]
    validate_bars(benchmark)
    spy = {row["date"]: row["adjusted_close"] for row in benchmark}
    dates = [row["date"] for row in observed]
    expected = [row["date"] for row in benchmark if dates[0] <= row["date"] <= dates[-1]]
    if dates != expected:
        raise ValueError("Observed sessions do not match the benchmark trading calendar")
    closes = np.array([row["adjusted_close"] for row in observed], dtype=float)
    volume = np.array([row["volume"] for row in observed], dtype=float)
    market = np.array([spy[day] for day in dates], dtype=float)
    returns = closes[1:] / closes[:-1] - 1
    market_returns = market[1:] / market[:-1] - 1
    values = (
        closes[-1] / closes[0] - 1,
        closes[-1] / closes[-6] - 1,
        np.std(returns, ddof=1), drawdown(closes), np.min(returns),
        np.mean(volume[-5:]) / np.mean(volume[:5]) - 1,
        market[-1] / market[0] - 1, np.std(market_returns, ddof=1),
    )
    return dict(zip(FEATURES, map(float, values)))


def build_dataset(bundle: dict) -> tuple[list[dict], list[dict]]:
    snapshots, exclusions = [], list(bundle["exclusions"])
    benchmark = bundle["benchmark"]["bars"]
    validate_bars(benchmark)
    for listing in bundle["listings"]:
        try:
            bars = listing["bars"][:OBSERVATION_SESSIONS + HORIZON_SESSIONS]
            if len(bars) < OBSERVATION_SESSIONS + HORIZON_SESSIONS:
                raise ValueError("Fewer than 40 complete sessions")
            validate_bars(bars)
            if bars[0]["date"] != listing["listing_date"]:
                raise ValueError("First observed price differs from the listing date")
            expected = [row["date"] for row in benchmark if bars[0]["date"] <= row["date"] <= bars[-1]["date"]]
            if expected != [row["date"] for row in bars]:
                raise ValueError("Missing sessions or trading-calendar mismatch")
            features = make_features(bars[:OBSERVATION_SESSIONS], benchmark)
            future = np.array([row["adjusted_close"] for row in bars[OBSERVATION_SESSIONS - 1:]], dtype=float)
            severity = drawdown(future)
            snapshots.append({
                "symbol": listing["symbol"], "sector": listing["sector"],
                "listing_date": listing["listing_date"],
                "as_of": bars[OBSERVATION_SESSIONS - 1]["date"],
                "label_end": bars[-1]["date"], "features": features,
                "drawdown": severity, "event": int(severity >= EVENT_THRESHOLD),
                "path": [{"date": row["date"], "value": row["adjusted_close"] / bars[19]["adjusted_close"] * 100} for row in bars],
                "source_url": listing["source_page"], "retrieved_at": listing["retrieved_at"],
            })
        except (ValueError, KeyError) as exc:
            exclusions.append({"symbol": listing["symbol"], "listing_date": listing["listing_date"], "reason": str(exc)})
    if len({row["symbol"] for row in snapshots}) != len(snapshots):
        raise ValueError("Duplicate issuer snapshots")
    return sorted(snapshots, key=lambda row: (row["as_of"], row["symbol"])), exclusions
