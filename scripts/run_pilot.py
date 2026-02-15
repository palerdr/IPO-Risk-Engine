"""
Stratified 120-symbol pilot: validate the expanded universe pipeline.

Samples 30 from 2010-2016, 45 from 2017-2021, 45 from 2022-2025.
Runs snapshot builder per symbol, logs audit table, reports gate criteria.

Gate criteria (must pass before full ingestion):
  - >= 70% of pilot symbols produce valid RIVER snapshots
  - No data leakage (forward labels don't leak into features)
  - Missing-data reasons logged for every failure
  - Runtime estimate extrapolated to full universe

Usage:
    python -m scripts.run_pilot
    python -m scripts.run_pilot --seed 42 --skip-existing
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.dataset.assemble import (
    snapshots_to_dataframe,
    temporal_train_val_test_split,
    save_dataset,
)
from ipo_risk_engine.snapshots.builder import (
    Snapshot,
    build_snapshots_for_symbol,
    prefetch_regime_bars,
)
from ipo_risk_engine.features.preflop_features import compute_preflop_features
from ipo_risk_engine.features.regime_features import get_sector_etf
from ipo_risk_engine.data.store import read_parquet, raw_bars_path

UNIVERSE_PATH = Path("data/ipo_universe.parquet")
PILOT_DIR = Path("data/pilot")

ERA_BINS = {
    "2010-2016": (2010, 2016, 30),
    "2017-2021": (2017, 2021, 45),
    "2022-2025": (2022, 2025, 45),
}

SECTOR_ETF_KEYS = [
    "technology", "health_care", "financials", "industrials",
    "consumer_goods", "consumer_services", "energy", "oil_gas",
    "real_estate",
]



def sic_to_sector(sic: str | None) -> str:
    """Map a 4-digit SIC code to a sector string matching SECTOR_ETF_MAP keys.

    Returns a sector string like 'technology', 'health_care', etc.
    Falls back to 'unknown' for unmapped codes.
    """
    if not sic or len(sic) < 2:
        return "unknown"

    div = int(sic[:2])

    if div <= 9:                          # agriculture, forestry, fishing
        return "unknown"
    if div <= 14:                         # mining, oil & gas extraction
        return "oil_gas" if sic.startswith("13") else "energy"
    if div <= 17:                         # construction
        return "industrials"
    if div <= 39:                         # manufacturing
        if sic.startswith("28"):          # chemicals, pharma
            return "health_care"
        if sic.startswith("35") or sic.startswith("36"):  # machinery, electronics
            return "technology"
        if sic.startswith("38"):          # instruments (medical + scientific)
            return "technology"
        if sic.startswith("37"):          # transportation equipment
            return "industrials"
        return "consumer_goods"           # food, textiles, paper, etc.
    if div <= 49:                         # transport, comms, utilities
        if sic.startswith("48"):          # communications
            return "technology"
        if sic.startswith("49"):          # electric/gas/water utilities
            return "energy"
        return "industrials"
    if div <= 51:                         # wholesale trade
        return "consumer_goods"
    if div <= 59:                         # retail trade
        return "consumer_services"
    if div <= 67:                         # finance, insurance, real estate
        if sic.startswith("65"):          # real estate
            return "real_estate"
        return "financials"
    if div <= 89:                         # services
        if sic.startswith("73"):          # business services (software)
            return "technology"
        if sic.startswith("80"):          # health services
            return "health_care"
        if sic.startswith("70") or sic.startswith("72"):
            return "consumer_services"    # hotels, personal services
        return "industrials"
    return "unknown"                      # 90-99 = public admin



def _filter_non_common(universe: pl.DataFrame) -> pl.DataFrame:
    """Exclude warrants, units, and preferred shares by ticker pattern."""
    return universe.filter(
        ~(
            (pl.col("symbol").str.len_chars() > 3) & pl.col("symbol").str.ends_with("W")
        ) & ~(
            (pl.col("symbol").str.len_chars() > 3) & pl.col("symbol").str.ends_with("U")
        ) & ~(
            pl.col("symbol").str.contains("-")
        )
    )


def stratified_sample(
    universe: pl.DataFrame,
    seed: int = 42,
) -> pl.DataFrame:
    """Sample symbols stratified by IPO era."""
    universe = _filter_non_common(universe)
    universe = universe.with_columns(
        pl.col("ipo_date").dt.year().alias("ipo_year")
    )

    sampled_frames = []
    rng = np.random.default_rng(seed)

    for era_name, (yr_start, yr_end, n_sample) in ERA_BINS.items():
        era_df = universe.filter(
            (pl.col("ipo_year") >= yr_start) & (pl.col("ipo_year") <= yr_end)
        )
        available = era_df.height
        actual_n = min(n_sample, available)

        if available == 0:
            print(f"  WARNING: No symbols in era {era_name}")
            continue

        indices = rng.choice(available, size=actual_n, replace=False)
        sampled = era_df[indices.tolist()]
        sampled_frames.append(sampled)
        print(f"  {era_name}: sampled {actual_n}/{available}")

    return pl.concat(sampled_frames).sort("ipo_date")



@dataclass
class AuditRow:
    symbol: str
    ipo_date: date
    era: str
    sector: str
    sic: str | None
    status: str           # "success" | "partial" | "failed"
    n_snapshots: int
    has_flop: bool
    has_turn: bool
    has_river: bool
    error: str | None
    elapsed_sec: float


def _era_label(yr: int) -> str:
    for name, (s, e, _) in ERA_BINS.items():
        if s <= yr <= e:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_dir: Path) -> set[str]:
    """Load set of already-processed symbols from checkpoint."""
    cp = checkpoint_dir / "checkpoint.json"
    if cp.exists():
        data = json.loads(cp.read_text())
        return set(data.get("processed_symbols", []))
    return set()


def save_checkpoint(processed: set[str], checkpoint_dir: Path) -> None:
    cp = checkpoint_dir / "checkpoint.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({
        "processed_symbols": sorted(processed),
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


# ---------------------------------------------------------------------------
# Gate criteria
# ---------------------------------------------------------------------------
def evaluate_gates(audit_rows: list[AuditRow], total_universe: int) -> dict:
    """Evaluate pilot gate criteria."""
    n = len(audit_rows)
    success = [r for r in audit_rows if r.status == "success"]
    with_river = [r for r in audit_rows if r.has_river]
    failed = [r for r in audit_rows if r.status == "failed"]

    river_rate = len(with_river) / n if n > 0 else 0.0
    avg_time = np.mean([r.elapsed_sec for r in audit_rows]) if audit_rows else 0
    estimated_full_hours = (avg_time * total_universe) / 3600

    # Failure reason breakdown
    failure_reasons: dict[str, int] = {}
    for r in audit_rows:
        if r.error:
            reason = r.error.split(":")[0].strip()
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    gates = {
        "pilot_n": n,
        "success_count": len(success),
        "river_count": len(with_river),
        "failed_count": len(failed),
        "river_rate": river_rate,
        "river_gate_pass": river_rate >= 0.70,
        "avg_sec_per_symbol": float(avg_time),
        "estimated_full_run_hours": float(estimated_full_hours),
        "total_universe": total_universe,
        "failure_reasons": failure_reasons,
    }

    # Per-era breakdown
    for era_name in ERA_BINS:
        era_rows = [r for r in audit_rows if r.era == era_name]
        era_river = [r for r in era_rows if r.has_river]
        gates[f"{era_name}_n"] = len(era_rows)
        gates[f"{era_name}_river_rate"] = (
            len(era_river) / len(era_rows) if era_rows else 0.0
        )

    return gates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified pilot or full dataset build")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume from checkpoint, skip already-processed symbols")
    parser.add_argument("--universe", type=Path, default=UNIVERSE_PATH,
                        help="Path to universe parquet (default: data/ipo_universe.parquet)")
    parser.add_argument("--full", action="store_true",
                        help="Use all symbols (no sampling). Writes to data/dataset/")
    args = parser.parse_args()

    output_dir = Path("data/dataset") if args.full else PILOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load universe
    print("=" * 60)
    mode_label = "FULL BUILD" if args.full else "STRATIFIED PILOT"
    print(f"{mode_label}: Loading universe")
    print("=" * 60)
    universe = pl.read_parquet(args.universe)
    print(f"  Universe: {universe.height} symbols")

    # 2. Sample or use all
    if args.full:
        pilot_df = _filter_non_common(universe)
        print(f"  Full build: {pilot_df.height} symbols (after non-common filter)")
    else:
        print("\nSampling 120 symbols by era...")
        pilot_df = stratified_sample(universe, seed=args.seed)
        print(f"  Pilot total: {pilot_df.height} symbols")

    # Save manifest
    pilot_df.write_parquet(output_dir / "pilot_symbols.parquet")

    # 3. Map SIC -> sector
    pilot_rows = pilot_df.to_dicts()
    for row in pilot_rows:
        row["sector"] = sic_to_sector(row.get("sic"))

    sector_dist = {}
    for row in pilot_rows:
        s = row["sector"]
        sector_dist[s] = sector_dist.get(s, 0) + 1
    print(f"\n  Sector distribution:")
    for s, c in sorted(sector_dist.items(), key=lambda x: -x[1]):
        print(f"    {s}: {c}")

    # 4. Pre-fetch regime bars
    print("\n" + "=" * 60)
    print("Pre-fetching regime bars (SPY + sector ETFs)")
    print("=" * 60)
    settings = load_settings()
    ipo_dates = [row["ipo_date"] for row in pilot_rows]
    sectors = [row["sector"] for row in pilot_rows]
    prefetch_regime_bars(ipo_dates, sectors, settings, force_refresh=True)

    # 5. Process symbols
    print("\n" + "=" * 60)
    print("Processing pilot symbols")
    print("=" * 60)

    # Load regime bars + universe for preflop features
    spy_bars_1d = read_parquet(raw_bars_path("SPY", "1d"))

    processed = load_checkpoint(output_dir) if args.skip_existing else set()
    if processed:
        print(f"  Resuming: {len(processed)} already processed")

    audit_rows: list[AuditRow] = []
    all_snapshots: list[Snapshot] = []

    for i, row in enumerate(pilot_rows):
        symbol = row["symbol"]
        ipo_d = row["ipo_date"]
        sector = row["sector"]
        sic = row.get("sic")
        yr = ipo_d.year

        if symbol in processed:
            continue

        print(f"\n  [{i+1}/{len(pilot_rows)}] {symbol} (IPO: {ipo_d}, sector: {sector})")

        # Compute PREFLOP features
        sector_etf = get_sector_etf(sector)
        sector_etf_bars = None
        if sector_etf:
            etf_path = raw_bars_path(sector_etf, "1d")
            if etf_path.exists():
                sector_etf_bars = read_parquet(etf_path)

        preflop = compute_preflop_features(
            ipo_date=ipo_d,
            sic=sic,
            filing_count=row.get("filing_count", 1),
            universe_df=universe,
            spy_bars_1d=spy_bars_1d,
            sector_etf_bars_1d=sector_etf_bars,
        )

        t0 = time.time()
        try:
            snaps = build_snapshots_for_symbol(
                symbol, ipo_d, sector, [7, 20], settings,
                preflop_features=preflop,
                force_refresh=True,
            )
            elapsed = time.time() - t0

            streets = {s.street for s in snaps}
            audit = AuditRow(
                symbol=symbol,
                ipo_date=ipo_d,
                era=_era_label(yr),
                sector=sector,
                sic=sic,
                status="success" if len(snaps) == 3 else "partial",
                n_snapshots=len(snaps),
                has_flop="FLOP" in streets,
                has_turn="TURN" in streets,
                has_river="RIVER" in streets,
                error=None if snaps else "no_snapshots_built",
                elapsed_sec=elapsed,
            )
            all_snapshots.extend(snaps)
            print(f"    -> {len(snaps)} snapshots ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            audit = AuditRow(
                symbol=symbol,
                ipo_date=ipo_d,
                era=_era_label(yr),
                sector=sector,
                sic=sic,
                status="failed",
                n_snapshots=0,
                has_flop=False,
                has_turn=False,
                has_river=False,
                error=str(e)[:200],
                elapsed_sec=elapsed,
            )
            print(f"    ERROR: {e}")

        audit_rows.append(audit)
        processed.add(symbol)

        # Checkpoint every 10 symbols
        if len(processed) % 10 == 0:
            save_checkpoint(processed, output_dir)

    # Final checkpoint
    save_checkpoint(processed, output_dir)

    # 6. Save audit table
    audit_dicts = [
        {
            "symbol": r.symbol,
            "ipo_date": r.ipo_date,
            "era": r.era,
            "sector": r.sector,
            "sic": r.sic,
            "status": r.status,
            "n_snapshots": r.n_snapshots,
            "has_flop": r.has_flop,
            "has_turn": r.has_turn,
            "has_river": r.has_river,
            "error": r.error or "",
            "elapsed_sec": r.elapsed_sec,
        }
        for r in audit_rows
    ]
    if audit_dicts:
        audit_path = output_dir / "ingestion_audit.parquet"
        pl.DataFrame(audit_dicts).write_parquet(audit_path)
        print(f"\n  Audit table: {audit_path} ({len(audit_dicts)} rows)")

    # 7. Build dataset if we have snapshots
    if all_snapshots:
        df = snapshots_to_dataframe(all_snapshots)
        save_dataset(df, output_dir / "snapshots_full.parquet")

        train_df, val_df, test_df = temporal_train_val_test_split(df)
        save_dataset(train_df, output_dir / "train.parquet")
        save_dataset(val_df, output_dir / "val.parquet")
        save_dataset(test_df, output_dir / "test.parquet")

        print(f"\n  Dataset: {df.height} rows "
              f"(train={train_df.height}, val={val_df.height}, test={test_df.height})")

    # 8. Gate criteria
    print("\n" + "=" * 60)
    print("GATE CRITERIA")
    print("=" * 60)

    gates = evaluate_gates(audit_rows, universe.height)

    print(f"  Pilot N:              {gates['pilot_n']}")
    print(f"  Success:              {gates['success_count']}")
    print(f"  With RIVER:           {gates['river_count']}")
    print(f"  Failed:               {gates['failed_count']}")
    print(f"  RIVER rate:           {gates['river_rate']:.1%}")
    print(f"  RIVER gate (>=70%):   {'PASS' if gates['river_gate_pass'] else 'FAIL'}")
    print(f"  Avg time/symbol:      {gates['avg_sec_per_symbol']:.1f}s")
    print(f"  Est. full run:        {gates['estimated_full_run_hours']:.1f}h "
          f"({gates['total_universe']} symbols)")

    print(f"\n  Per-era RIVER rates:")
    for era_name in ERA_BINS:
        n = gates.get(f"{era_name}_n", 0)
        rate = gates.get(f"{era_name}_river_rate", 0)
        print(f"    {era_name}: {rate:.1%} ({n} symbols)")

    if gates["failure_reasons"]:
        print(f"\n  Failure reasons:")
        for reason, count in sorted(
            gates["failure_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"    {reason}: {count}")

    # Save gate results
    gate_path = output_dir / "gate_results.json"
    with open(gate_path, "w") as f:
        json.dump(gates, f, indent=2, default=str)
    print(f"\n  Gate results: {gate_path}")


if __name__ == "__main__":
    main()
