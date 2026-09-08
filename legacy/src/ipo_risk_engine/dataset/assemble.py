"""
Dataset assembly: convert Snapshots to a flat Polars DataFrame + temporal splits.
"""
from __future__ import annotations
from pathlib import Path
import polars as pl
from ipo_risk_engine.snapshots.builder import Snapshot


def snapshots_to_dataframe(snapshots: list[Snapshot]) -> pl.DataFrame:
    """Flatten snapshots into a single DataFrame with derived labels."""
    rows = []
    for snap in snapshots:
        row: dict = {
            "symbol": snap.symbol,
            "street": snap.street,
            "asof_date": snap.asof_date,
            "ipo_date": snap.ipo_date,
            "sector": snap.sector,
            **snap.features,
            **snap.labels,
            **snap.quality_flags,
        }
        mdd = snap.labels.get("forward_mdd_20d")
        if mdd is not None:
            severity = -mdd
            row["risk_severity"] = severity
            row["adverse_20"] = 1 if severity >= 0.20 else 0
            row["severe_30"] = 1 if severity >= 0.30 else 0
        else:
            row["risk_severity"] = None
            row["adverse_20"] = None
            row["severe_30"] = None
        rows.append(row)
    return pl.DataFrame(rows)


def temporal_train_val_test_split(
    df: pl.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split by IPO chronology while keeping each symbol in one split."""
    
    unique = df.select(["symbol", "asof_date"]).group_by("symbol").agg(pl.col("asof_date").min()).sort("asof_date")
    n = unique.height
    train_cutoff = int(n*train_frac)
    val_cutoff = int(n*(train_frac + val_frac))
    train_symbols = unique[:train_cutoff]["symbol"].to_list()
    val_symbols = unique[train_cutoff:val_cutoff]["symbol"].to_list()
    test_symbols = unique[val_cutoff:]["symbol"].to_list()
    train_df = df.filter(pl.col("symbol").is_in(train_symbols))
    val_df = df.filter(pl.col("symbol").is_in(val_symbols))
    test_df = df.filter(pl.col("symbol").is_in(test_symbols))
    return train_df, val_df, test_df
    


def save_dataset(df: pl.DataFrame, path: Path) -> None:  
    """Save dataset DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"Saved {df.height} rows to {path}")
