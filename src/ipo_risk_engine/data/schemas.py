"""
Schema definitions and validation for stock bar data
"""

import polars as pl

BASE_BAR_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us", "UTC"),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64()
  }

# Optional columns that may be present (vwap, trade_count)
OPTIONAL_BAR_COLUMNS: dict[str, pl.DataType] = {
      "vwap": pl.Float64(),
      "trade_count": pl.Int64()
  }

def _validate_schema(df:pl.DataFrame, expected: dict[str, pl.DataType], name:str):
    schema = df.schema
    missing = [c for c in expected if c not in schema]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    
    mismatched = []

    for col, exp_dtype in expected.items():
        got_dtype = schema[col]
        if got_dtype != exp_dtype:
            mismatched.append((col, exp_dtype, got_dtype))
        
    if mismatched:
        msg_lines = [f"{name}: dtype mismatches:"]
        for col, exp, got in mismatched:
            msg_lines.append(f"  - {col}: expected {exp}, got {got}")
        raise TypeError("\n".join(msg_lines))
    
def validate_bars(df: pl.DataFrame, timeframe: str) -> None:
    return _validate_schema(df, BASE_BAR_SCHEMA, timeframe)