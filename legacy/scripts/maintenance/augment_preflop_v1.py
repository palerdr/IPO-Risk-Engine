"""
Augment snapshots_full.parquet with PREFLOP V1 structured features.

Merges:
  1. Preflop metadata (s1_lead_days, s1a_count, days_since_last_amendment,
     exchange_nyse, exchange_nasdaq) — joined by symbol
  2. Sector one-hot encoding — derived from existing 'sector' column

Null handling:
  - s1_lead_days: fill with median (reasonable default for missing bar data)
  - s1a_count: fill with 0 (assume no amendments if not found)
  - days_since_last_amendment: fill with -1 (sentinel for "no amendment")
  - Sector one-hot: all 0 if sector is "unknown"

Output:
  data/dataset/snapshots_full.parquet  (overwritten with new columns)

Usage:
    python -m scripts.augment_preflop_v1
"""
import polars as pl
from pathlib import Path

SNAPSHOT_PATH = Path("data/dataset/snapshots_full.parquet")
METADATA_PATH = Path("data/features/preflop_metadata.parquet")

# Sectors for one-hot (drop "unknown" as reference category)
KNOWN_SECTORS = [
    "health_care", "technology", "industrials", "financials",
    "consumer_goods", "consumer_services", "energy", "oil_gas", "real_estate",
]


def main() -> None:
    print("=" * 60)
    print("AUGMENT SNAPSHOTS WITH PREFLOP V1 FEATURES")
    print("=" * 60)

    snaps = pl.read_parquet(SNAPSHOT_PATH)
    meta = pl.read_parquet(METADATA_PATH)
    print(f"  Snapshots: {snaps.shape} ({snaps.columns})")
    print(f"  Metadata:  {meta.shape}")

    # Select only the feature columns from metadata (not raw dates)
    # Deduplicate by symbol (keep first row per symbol)
    meta_features = (
        meta.sort("s1_filing_date")
        .group_by("symbol")
        .first()
        .select([
            "symbol",
            "preflop_s1_lead_days",
            "preflop_exchange_nyse",
            "preflop_exchange_nasdaq",
            "preflop_s1a_count",
            "preflop_days_since_last_amendment",
        ])
    )
    print(f"  Metadata (deduped): {meta_features.height} symbols")

    # Drop any existing preflop_v1 columns (idempotent re-run)
    # CAREFUL: don't match preflop_sector_ipo_heat_90d or preflop_sector_return_20d
    _SAFE_EXISTING = {"preflop_sector_ipo_heat_90d", "preflop_sector_return_20d"}
    existing_v1 = [c for c in snaps.columns
                   if (c.startswith("preflop_s1")
                       or c.startswith("preflop_exchange")
                       or c.startswith("preflop_sector_")
                       or c == "preflop_days_since_last_amendment")
                   and c not in _SAFE_EXISTING]
    if existing_v1:
        snaps = snaps.drop(existing_v1)
        print(f"  Dropped {len(existing_v1)} existing v1 columns (re-run)")

    # Join by symbol
    augmented = snaps.join(meta_features, on="symbol", how="left")

    # Fill nulls with sensible defaults
    median_lead = float(augmented["preflop_s1_lead_days"].median() or 0.0)  # type: ignore[arg-type]
    augmented = augmented.with_columns([
        pl.col("preflop_s1_lead_days").fill_null(median_lead),
        pl.col("preflop_exchange_nyse").fill_null(0.0),
        pl.col("preflop_exchange_nasdaq").fill_null(0.0),
        pl.col("preflop_s1a_count").fill_null(0.0),
        pl.col("preflop_days_since_last_amendment").fill_null(-1.0),
    ])

    # Add sector one-hot encoding
    for sector in KNOWN_SECTORS:
        col_name = f"preflop_sector_{sector}"
        augmented = augmented.with_columns(
            (pl.col("sector") == sector).cast(pl.Float64).alias(col_name)
        )

    # Report
    new_cols = [c for c in augmented.columns if c not in snaps.columns]
    print(f"\n  New columns ({len(new_cols)}):")
    for c in new_cols:
        nulls = augmented[c].null_count()
        print(f"    {c}: nulls={nulls}, mean={augmented[c].mean():.3f}")

    # Save
    augmented.write_parquet(SNAPSHOT_PATH)
    print(f"\n  Saved: {SNAPSHOT_PATH} ({augmented.shape})")
    print(f"  Total columns: {len(augmented.columns)}")


if __name__ == "__main__":
    main()
