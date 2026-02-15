"""
Snapshot builder: assembles (features + labels) per street per symbol.

Each snapshot is one row of training data (all streets use daily bars):
- FLOP snapshot:  11 features + regime + labels
- TURN snapshot:  22 features (FLOP + TURN) + regime + labels
- RIVER snapshot: 33 features (FLOP + TURN + RIVER) + regime + labels

Labels are forward MDD and MFR at configurable horizons,
computed from daily bars starting the day after the street boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ipo_risk_engine.config.settings import load_settings, AlpacaSettings as Settings
from ipo_risk_engine.data.ingest import ingest_bars
from ipo_risk_engine.data.store import read_parquet, raw_bars_path
from ipo_risk_engine.features.streets import (
    Street,
    StreetWindow,
    compute_street_windows,
)
from ipo_risk_engine.features.street_features import compute_daily_street_features
from ipo_risk_engine.features.regime_features import (
    compute_regime_features,
    get_sector_etf,
)
from ipo_risk_engine.labels.mdd import compute_forward_mdd, compute_forward_max_runup


@dataclass
class Snapshot:
    symbol: str
    street: Street
    asof_date: date
    sector: str
    ipo_date: date | None
    features: dict[str, float]
    labels: dict[str, float | None]
    quality_flags: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IPOEntry:
    ipo_date: date
    sector: str


# --- IPO registry (94 symbols, 2024-2025) ---

IPO_REGISTRY: dict[str, IPOEntry] = {
    # ── 2024 Q1 ──
    "SDHC": IPOEntry(date(2024, 1, 11), "consumer_goods"),
    "KSPI": IPOEntry(date(2024, 1, 19), "financials"),
    "CGON": IPOEntry(date(2024, 1, 25), "health_care"),
    "BTSG": IPOEntry(date(2024, 1, 26), "health_care"),
    "AS":   IPOEntry(date(2024, 2, 1),  "consumer_goods"),
    "AHR":  IPOEntry(date(2024, 2, 7),  "real_estate"),
    "KYTX": IPOEntry(date(2024, 2, 8),  "health_care"),
    "MGX":  IPOEntry(date(2024, 2, 9),  "health_care"),
    "TBBB": IPOEntry(date(2024, 2, 9),  "consumer_goods"),
    "ALAB": IPOEntry(date(2024, 3, 20), "technology"),
    "RDDT": IPOEntry(date(2024, 3, 21), "technology"),
    # ── 2024 Q2 ──
    "CTNM": IPOEntry(date(2024, 4, 5),  "health_care"),
    "PACS": IPOEntry(date(2024, 4, 11), "health_care"),
    "ULS":  IPOEntry(date(2024, 4, 12), "industrials"),
    "IBTA": IPOEntry(date(2024, 4, 18), "technology"),
    "CTRI": IPOEntry(date(2024, 4, 18), "industrials"),
    "LOAR": IPOEntry(date(2024, 4, 25), "industrials"),
    "RBRK": IPOEntry(date(2024, 4, 25), "technology"),
    "MRX":  IPOEntry(date(2024, 4, 25), "financials"),
    "VIK":  IPOEntry(date(2024, 5, 1),  "consumer_services"),
    "NNE":  IPOEntry(date(2024, 5, 8),  "energy"),
    "PAL":  IPOEntry(date(2024, 5, 9),  "industrials"),
    "SVCO": IPOEntry(date(2024, 5, 9),  "technology"),
    "ZK":   IPOEntry(date(2024, 5, 10), "technology"),
    "HDL":  IPOEntry(date(2024, 5, 17), "consumer_services"),
    "BOW":  IPOEntry(date(2024, 5, 23), "financials"),
    "FLYE": IPOEntry(date(2024, 6, 6),  "consumer_goods"),
    "GAUZ": IPOEntry(date(2024, 6, 6),  "technology"),
    "WAY":  IPOEntry(date(2024, 6, 7),  "technology"),
    "RAPP": IPOEntry(date(2024, 6, 7),  "health_care"),
    "TEM":  IPOEntry(date(2024, 6, 14), "technology"),
    "WBTN": IPOEntry(date(2024, 6, 27), "technology"),
    "ALMS": IPOEntry(date(2024, 6, 28), "health_care"),
    "LB":   IPOEntry(date(2024, 6, 28), "oil_gas"),
    # ── 2024 Q3 ──
    "ARDT": IPOEntry(date(2024, 7, 18), "health_care"),
    "TWFG": IPOEntry(date(2024, 7, 18), "financials"),
    "ARTV": IPOEntry(date(2024, 7, 19), "health_care"),
    "OS":   IPOEntry(date(2024, 7, 24), "technology"),
    "LINE": IPOEntry(date(2024, 7, 25), "industrials"),
    "CON":  IPOEntry(date(2024, 7, 25), "health_care"),
    "ACTU": IPOEntry(date(2024, 8, 13), "health_care"),
    "MBX":  IPOEntry(date(2024, 9, 13), "health_care"),
    "BCAX": IPOEntry(date(2024, 9, 13), "health_care"),
    "ZBIO": IPOEntry(date(2024, 9, 13), "health_care"),
    "BKV":  IPOEntry(date(2024, 9, 26), "oil_gas"),
    "BIOA": IPOEntry(date(2024, 9, 26), "health_care"),
    "GRDN": IPOEntry(date(2024, 9, 26), "health_care"),
    # ── 2024 Q4 ──
    "SARO": IPOEntry(date(2024, 10, 2),  "industrials"),
    "FVR":  IPOEntry(date(2024, 10, 2),  "real_estate"),
    "CBNA": IPOEntry(date(2024, 10, 4),  "financials"),
    "KLC":  IPOEntry(date(2024, 10, 9),  "consumer_services"),
    "CAMP": IPOEntry(date(2024, 10, 11), "health_care"),
    "CBLL": IPOEntry(date(2024, 10, 11), "health_care"),
    "UPB":  IPOEntry(date(2024, 10, 11), "health_care"),
    "INGM": IPOEntry(date(2024, 10, 24), "technology"),
    "SEPN": IPOEntry(date(2024, 10, 25), "health_care"),
    "WRD":  IPOEntry(date(2024, 10, 25), "technology"),
    "GELS": IPOEntry(date(2024, 10, 29), "health_care"),
    "PLRZ": IPOEntry(date(2024, 10, 29), "health_care"),
    "PONY": IPOEntry(date(2024, 11, 27), "technology"),
    "TTAN": IPOEntry(date(2024, 12, 12), "technology"),
    "AVR":  IPOEntry(date(2024, 12, 13), "health_care"),
    # ── 2025 Q1-Q2 ──
    "CRWV": IPOEntry(date(2025, 3, 28), "technology"),
    "CRCL": IPOEntry(date(2025, 6, 5),  "financials"),
    "ETOR": IPOEntry(date(2025, 6, 10), "technology"),
    "CHYM": IPOEntry(date(2025, 6, 12), "health_care"),
    "VOYG": IPOEntry(date(2025, 6, 12), "technology"),
    # ── 2025 Q3 ──
    "CARL": IPOEntry(date(2025, 7, 23), "health_care"),
    "NIQ":  IPOEntry(date(2025, 7, 23), "technology"),
    "ARX":  IPOEntry(date(2025, 7, 24), "financials"),
    "MH":   IPOEntry(date(2025, 7, 24), "consumer_services"),
    "AMBQ": IPOEntry(date(2025, 7, 30), "technology"),
    "FIG":  IPOEntry(date(2025, 7, 31), "technology"),
    "SI":   IPOEntry(date(2025, 7, 31), "health_care"),
    "HTFL": IPOEntry(date(2025, 8, 8),  "health_care"),
    "DKI":  IPOEntry(date(2025, 8, 8),  "technology"),
    "BLSH": IPOEntry(date(2025, 8, 13), "financials"),
    "MIAX": IPOEntry(date(2025, 8, 14), "financials"),
    "NSRX": IPOEntry(date(2025, 8, 14), "health_care"),
    "CURX": IPOEntry(date(2025, 8, 26), "health_care"),
    "PMI":  IPOEntry(date(2025, 8, 29), "health_care"),
    "FCHL": IPOEntry(date(2025, 9, 4),  "consumer_services"),
    "KLAR": IPOEntry(date(2025, 9, 10), "financials"),
    "FIGR": IPOEntry(date(2025, 9, 11), "technology"),
    "LBRX": IPOEntry(date(2025, 9, 11), "health_care"),
    "BRCB": IPOEntry(date(2025, 9, 12), "consumer_services"),
    "LGN":  IPOEntry(date(2025, 9, 12), "industrials"),
    "VIA":  IPOEntry(date(2025, 9, 12), "technology"),
    "STUB": IPOEntry(date(2025, 9, 17), "consumer_services"),
    "WBI":  IPOEntry(date(2025, 9, 17), "oil_gas"),
    "NTSK": IPOEntry(date(2025, 9, 18), "technology"),
    "PTRN": IPOEntry(date(2025, 9, 19), "consumer_goods"),
    "CBK":  IPOEntry(date(2025, 10, 2), "financials"),
    "ALH":  IPOEntry(date(2025, 10, 9), "industrials"),
}


# --- Ingestion helpers ---

def _ingest_daily_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    settings: Settings,
    *,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Ingest only daily bars for a symbol and return the DataFrame."""
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    ingest_bars(symbol, start, end, tf_1d, "1d", settings=settings,
                force_refresh=force_refresh)
    return read_parquet(raw_bars_path(symbol, "1d"))


def _get_trading_days(bars_1d: pl.DataFrame) -> list[date]:
    """Extract sorted list of trading dates from daily bars."""
    return (
        bars_1d.select(pl.col("ts").dt.date().alias("d"))
        .unique()
        .sort("d")["d"]
        .to_list()
    )


def _filter_bars_to_window(
    bars: pl.DataFrame, window: StreetWindow
) -> pl.DataFrame:
    """Filter bars to a street window's [start, end) range."""
    return bars.filter(
        (pl.col("ts") >= window.start) & (pl.col("ts") < window.end)
    )


def _compute_quality_flags(bars_1d: pl.DataFrame) -> dict[str, float]:
    """Compute data quality metrics from daily bars."""
    zero_vol = bars_1d.select((pl.col("volume") == 0).mean()).item()
    zero_volume_pct = float(zero_vol * 100) if zero_vol is not None else 0.0
    med = bars_1d["volume"].median()
    median_daily_volume = float(med) if med is not None else 0.0

    return {
        "bar_count_1d": float(bars_1d.height),
        "zero_volume_pct": zero_volume_pct,
        "median_daily_volume": median_daily_volume,
    }


# --- Core builder ---

def build_snapshots_for_symbol(
    symbol: str,
    ipo_date: date,
    sector: str,
    horizons: list[int],
    settings: Settings,
    *,
    preflop_features: dict[str, float] | None = None,
    force_refresh: bool = False,
) -> list[Snapshot]:
    """
    Build snapshots for all streets of a single symbol.

    All streets use daily bars only. Returns a list of Snapshot objects
    (one per street that has enough data).
    Features accumulate: TURN includes FLOP features, RIVER includes both.
    If preflop_features is provided, they are included in all snapshots.
    """
    max_horizon = max(horizons)
    start = datetime.combine(ipo_date, datetime.min.time())
    # 61 trading days (~86 cal days) for streets + max_horizon trading days for labels
    # Use 130 + max_horizon to ensure RIVER rows get forward labels even with holidays
    end = start + timedelta(days=130 + max_horizon)

    print(f"  Ingesting {symbol} daily bars ({ipo_date})...")
    bars_1d = _ingest_daily_bars(symbol, start, end, settings,
                                  force_refresh=force_refresh)

    # Read regime bars from pre-fetched cache
    spy_bars_1d = read_parquet(raw_bars_path("SPY", "1d"))
    sector_etf = get_sector_etf(sector)
    sector_etf_bars_1d = None
    if sector_etf:
        etf_path = raw_bars_path(sector_etf, "1d")
        if etf_path.exists():
            sector_etf_bars_1d = read_parquet(etf_path)

    # Get trading days and compute street windows
    trading_days = _get_trading_days(bars_1d)
    if not trading_days or trading_days[0] != ipo_date:
        trading_days = [d for d in trading_days if d >= ipo_date]
        if not trading_days:
            print(f"  WARNING: No trading days found for {symbol} after {ipo_date}")
            return []

    windows = compute_street_windows(
        listing_day=trading_days[0],
        trading_days=trading_days,
    )

    # Precompute forward labels from daily bars
    label_dfs = {}
    for h in horizons:
        mdd_df = compute_forward_mdd(bars_1d, horizon=h)
        mfr_df = compute_forward_max_runup(bars_1d, horizon=h)
        label_dfs[h] = (mdd_df, mfr_df)

    # Quality flags from all daily bars
    quality_flags = _compute_quality_flags(bars_1d)

    snapshots: list[Snapshot] = []
    accumulated_features: dict[str, float] = {}

    # Seed with preflop features if provided
    if preflop_features:
        accumulated_features.update(preflop_features)

    for window in windows:
        street = window.street
        as_of = (window.end - timedelta(days=1)).date()

        street_bars = _filter_bars_to_window(bars_1d, window=window)
        features = compute_daily_street_features(street_bars, street)

        # Skip street if all features are None (not enough bars)
        if all(v is None for v in features.values()):
            print(f"    Not enough bars for {street}")
            continue

        accumulated_features.update(features)

        # Regime/context features
        regime_spy = _filter_bars_to_window(spy_bars_1d, window)
        regime_sector = (
            _filter_bars_to_window(sector_etf_bars_1d, window)
            if sector_etf_bars_1d is not None else None
        )
        regime_feats = compute_regime_features(
            regime_spy, regime_sector, street, ipo_date
        )
        accumulated_features.update(regime_feats)

        snap = Snapshot(
            symbol=symbol, street=street, asof_date=as_of, sector=sector,
            ipo_date=ipo_date, features=dict(accumulated_features),
            labels={}, quality_flags=quality_flags,
        )

        # Forward labels
        labels = {}
        for h in horizons:
            mdd_df, mfr_df = label_dfs[h]
            mdd_row = mdd_df.filter(pl.col("ts").dt.date() == as_of)
            mfr_row = mfr_df.filter(pl.col("ts").dt.date() == as_of)

            labels[f"forward_mdd_{h}d"] = (
                mdd_row[f"forward_mdd_{h}d"][0] if mdd_row.height > 0 else None
            )
            labels[f"forward_mfr_{h}d"] = (
                mfr_row[f"forward_mfr_{h}d"][0] if mfr_row.height > 0 else None
            )
        snap.labels.update(labels)
        snapshots.append(snap)

    return snapshots


def prefetch_regime_bars(
    ipo_dates: list[date],
    sectors: list[str],
    settings: Settings,
    *,
    force_refresh: bool = False,
) -> None:
    """Pre-fetch SPY + sector ETF daily bars covering all IPO dates.

    Must be called before build_snapshots_for_symbol so the cache contains
    the full date range needed.
    """
    tf_1d = TimeFrame(1, TimeFrameUnit.Day)
    earliest = min(ipo_dates)
    # Buffer: 30 days before earliest IPO, 90 days after latest
    start = datetime(earliest.year, 1, 1)
    end = datetime.now()

    # SPY
    print(f"  Pre-fetching SPY daily bars ({start.date()} -> {end.date()})...")
    ingest_bars("SPY", start, end, tf_1d, "1d", settings=settings,
                force_refresh=force_refresh)

    # Sector ETFs (deduplicated)
    etfs_needed = set()
    for sector in sectors:
        etf = get_sector_etf(sector)
        if etf:
            etfs_needed.add(etf)

    for etf in sorted(etfs_needed):
        print(f"  Pre-fetching {etf} daily bars...")
        try:
            ingest_bars(etf, start, end, tf_1d, "1d", settings=settings,
                        force_refresh=force_refresh)
        except Exception as e:
            print(f"  WARNING: Could not fetch {etf}: {e}")


def build_all_snapshots(
    horizons: list[int] | None = None,
    symbols: dict[str, IPOEntry] | None = None,
    universe_path: Path | None = None,
) -> list[Snapshot]:
    """Build snapshots for all symbols.

    If universe_path is provided, loads the EDGAR parquet and computes
    PREFLOP features. Otherwise falls back to IPO_REGISTRY without PREFLOP.
    """
    from ipo_risk_engine.features.preflop_features import compute_preflop_features

    if horizons is None:
        horizons = [7, 20]

    universe_df = None
    if universe_path and universe_path.exists():
        universe_df = pl.read_parquet(universe_path)
        rows = universe_df.to_dicts()
        symbols = {r["symbol"]: IPOEntry(r["ipo_date"], "unknown") for r in rows}
    elif symbols is None:
        symbols = IPO_REGISTRY

    settings = load_settings()

    ipo_dates = [e.ipo_date for e in symbols.values()]
    sectors = [e.sector for e in symbols.values()]
    prefetch_regime_bars(ipo_dates, sectors, settings, force_refresh=True)

    spy_bars_1d = read_parquet(raw_bars_path("SPY", "1d"))
    all_snapshots: list[Snapshot] = []

    for symbol, entry in symbols.items():
        print(f"\nBuilding snapshots for {symbol} (IPO: {entry.ipo_date})")

        preflop = None
        if universe_df is not None:
            sym_row = universe_df.filter(pl.col("symbol") == symbol)
            sic = sym_row["sic"][0] if sym_row.height > 0 and "sic" in sym_row.columns else None
            fc = int(sym_row["filing_count"][0]) if sym_row.height > 0 and "filing_count" in sym_row.columns else 1

            etf_sym = get_sector_etf(entry.sector)
            sector_etf_bars = None
            if etf_sym:
                etf_path = raw_bars_path(etf_sym, "1d")
                if etf_path.exists():
                    sector_etf_bars = read_parquet(etf_path)

            preflop = compute_preflop_features(
                ipo_date=entry.ipo_date, sic=sic, filing_count=fc,
                universe_df=universe_df, spy_bars_1d=spy_bars_1d,
                sector_etf_bars_1d=sector_etf_bars,
            )

        try:
            snaps = build_snapshots_for_symbol(
                symbol, entry.ipo_date, entry.sector, horizons, settings,
                preflop_features=preflop,
            )
            all_snapshots.extend(snaps)
            print(f"  -> {len(snaps)} snapshots built")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nTotal snapshots: {len(all_snapshots)}")
    return all_snapshots
