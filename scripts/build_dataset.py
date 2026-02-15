"""Build snapshots dataset and temporal train/val/test splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from ipo_risk_engine.snapshots.builder import build_all_snapshots, IPO_REGISTRY
from ipo_risk_engine.dataset.assemble import (
    snapshots_to_dataframe,
    temporal_train_val_test_split,
    save_dataset,
)

DATASET_DIR = Path("data/dataset")


def main():
    parser = argparse.ArgumentParser(description="Build IPO dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N symbols (for testing)")
    args = parser.parse_args()

    symbols = IPO_REGISTRY
    if args.limit:
        symbols = dict(list(IPO_REGISTRY.items())[:args.limit])
        print(f"Limited to {len(symbols)} symbols for testing")

    print(f"\n{'='*60}")
    print(f"Building snapshots for {len(symbols)} symbols...")
    print(f"{'='*60}")
    snapshots = build_all_snapshots(horizons=[7, 20], symbols=symbols)

    if not snapshots:
        print("No snapshots built. Exiting.")
        return

    df = snapshots_to_dataframe(snapshots)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns}")

    save_dataset(df, DATASET_DIR / "snapshots_full.parquet")

    train_df, val_df, test_df = temporal_train_val_test_split(df)

    print(f"\n{'='*60}")
    print("Temporal Split Summary:")
    print(f"  Train: {train_df.height} rows ({train_df['symbol'].n_unique()} symbols)")
    print(f"  Val:   {val_df.height} rows ({val_df['symbol'].n_unique()} symbols)")
    print(f"  Test:  {test_df.height} rows ({test_df['symbol'].n_unique()} symbols)")

    save_dataset(train_df, DATASET_DIR / "train.parquet")
    save_dataset(val_df, DATASET_DIR / "val.parquet")
    save_dataset(test_df, DATASET_DIR / "test.parquet")

    print(f"\n{'='*60}")
    print("Event Base Rates by Street:")
    for label, desc in [("adverse_20", "tau=0.20"), ("severe_30", "tau=0.30")]:
        print(f"\n  {label} ({desc}):")
        for street in df["street"].unique().sort().to_list():
            street_df = df.filter(
                (pl.col("street") == street) & pl.col(label).is_not_null()
            )
            n = street_df.height
            if n > 0:
                n_pos = int(street_df[label].sum())
                print(f"    {street}: {n_pos}/{n} = {n_pos/n:.1%}")
            else:
                print(f"    {street}: no labeled rows")

    print(f"\n{'='*60}")
    print("Sanity Checks:")

    train_syms = set(train_df["symbol"].unique().to_list())
    val_syms = set(val_df["symbol"].unique().to_list())
    test_syms = set(test_df["symbol"].unique().to_list())
    assert train_syms.isdisjoint(val_syms), "Train/val symbol overlap!"
    assert train_syms.isdisjoint(test_syms), "Train/test symbol overlap!"
    assert val_syms.isdisjoint(test_syms), "Val/test symbol overlap!"
    print("  No symbol overlap across splits: PASS")

    assert train_df.height + val_df.height + test_df.height == df.height
    print("  Row counts sum correctly: PASS")

    train_max = train_df["asof_date"].max()
    val_min = val_df["asof_date"].min()
    test_min = test_df["asof_date"].min()
    print(f"  Train max asof: {train_max}")
    print(f"  Val   min asof: {val_min}")
    print(f"  Test  min asof: {test_min}")

    print(f"\n{'='*60}")
    print("Dataset build complete!")


if __name__ == "__main__":
    main()
