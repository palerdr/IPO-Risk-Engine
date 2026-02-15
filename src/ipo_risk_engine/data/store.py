from __future__ import annotations

from pathlib import Path
import polars as pl

DATA_DIR = Path("data")


def raw_symbol_dir(symbol: str) -> Path:
    return DATA_DIR / "raw" / symbol.upper()


def raw_bars_path(symbol: str, timeframe_key: str) -> Path:
    return raw_symbol_dir(symbol) / f"bars_{timeframe_key}.parquet"


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)
