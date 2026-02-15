from datetime import datetime, timezone

import numpy as np
import polars as pl

from ipo_risk_engine.features.snapshots import build_snapshot_table
from ipo_risk_engine.policy.retrieval import estimate_tail_risk, fit_retrieval_index, query


def _dt(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_snapshot_table_columns(tmp_path) -> None:
    bars = pl.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "ts": [
                _dt(2020, 9, 1),
                _dt(2020, 9, 2),
                _dt(2020, 9, 3),
                _dt(2020, 9, 4),
                _dt(2020, 9, 5),
                _dt(2020, 9, 6),
            ],
            "open": [10.0, 10.5, 11.0, 10.8, 10.2, 10.1],
            "high": [10.2, 10.7, 11.2, 11.0, 10.4, 10.3],
            "low": [9.8, 10.3, 10.8, 10.5, 10.0, 10.0],
            "close": [10.1, 10.6, 10.9, 10.7, 10.1, 10.2],
            "volume": [100, 120, 130, 110, 115, 100],
        }
    )

    path = tmp_path / "raw" / "AAA" / "bars_1d.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(path)
    snapshot_path = tmp_path / "snapshots.parquet"

    table = build_snapshot_table(
        ["AAA"],
        data_dir=tmp_path,
        flop_days=1,
        turn_days=1,
        river_days=1,
        write_out=True,
        output_path=snapshot_path,
    )

    assert "symbol" in table.columns
    assert "street" in table.columns
    assert "asof" in table.columns
    assert "y_1" in table.columns
    assert table.height >= 1


def test_retrieval_excludes_self() -> None:
    table = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "street": ["FLOP", "FLOP", "FLOP"],
            "asof": [_dt(2020, 9, 1), _dt(2020, 9, 2), _dt(2020, 9, 3)],
            "f1": [0.0, 1.0, 2.0],
            "f2": [0.0, 1.0, 2.0],
            "y_1": [0.0, 1.0, 0.0],
            "y_3": [0.0, 0.0, 1.0],
        }
    )
    index = fit_retrieval_index(table, ["f1", "f2"])
    row = table.row(0, named=True)
    comps = query(index, "FLOP", row, k=2, exclude_id=("AAA", "FLOP", row["asof"]))
    assert comps.height == 2
    assert "AAA" not in comps.get_column("symbol").to_list()


def test_estimate_tail_risk_mean() -> None:
    comps = pl.DataFrame(
        {
            "distance": [1.0, 2.0, 3.0],
            "y_1": [0.0, 1.0, 1.0],
            "y_3": [1.0, 1.0, 0.0],
        }
    )
    p_hat, _ = estimate_tail_risk(comps, horizons=(1, 3), weighted=False)
    assert p_hat["y_1"] == np.mean([0.0, 1.0, 1.0])
    assert p_hat["y_3"] == np.mean([1.0, 1.0, 0.0])
