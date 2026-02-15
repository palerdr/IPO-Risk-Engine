"""
Define the expected column names and types for events and labels
"""
import polars as pl

EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us"),
    "event_type": pl.String(),
    "side": pl.String(),
    "size": pl.Float64(),
    "price": pl.Float64(),
    "source": pl.String(),
}

LABELS_SCHEMA: dict[str, pl.DataType] = {
    "ts": pl.Datetime("us"),
    "regime_id": pl.Int64(),
    "vol": pl.Float64(),
    "liq": pl.Float64(),
}

def _validate_schema(lf:pl.LazyFrame, expected: dict[str, pl.DataType], name:str):
    schema = lf.collect_schema()
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


def validate_events(lf: pl.LazyFrame) -> None:
    return _validate_schema(lf, EVENTS_SCHEMA, "events")


def validate_labels(lf: pl.LazyFrame) ->  None:
    return _validate_schema(lf, LABELS_SCHEMA, "labels")
