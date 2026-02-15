from __future__ import annotations

import polars as pl

from ipo_risk_engine.features.snapshots import SNAPSHOT_PATH
from ipo_risk_engine.policy.retrieval import estimate_tail_risk, fit_retrieval_index, query


def main() -> None:
    table = pl.read_parquet(SNAPSHOT_PATH)
    feature_cols = [
        c
        for c in table.columns
        if c
        not in {
            "symbol",
            "street",
            "asof",
            "y_1",
            "y_3",
            "y_5",
            "y_10",
        }
    ]
    index = fit_retrieval_index(table, feature_cols)

    pltr = table.filter(pl.col("symbol") == "PLTR").sort("asof")
    if pltr.is_empty():
        raise ValueError("No PLTR rows found in snapshot table")

    row = pltr.row(0, named=True)
    exclude_id = (row["symbol"], row["street"], row["asof"])

    comps = query(index, row["street"], row, k=10, exclude_id=exclude_id)
    p_hat, diagnostics = estimate_tail_risk(comps, weighted=True)

    print("p_hat:", p_hat)
    print("diagnostics:", diagnostics)
    print("top comps:")
    print(comps.head(5))


if __name__ == "__main__":
    main()
