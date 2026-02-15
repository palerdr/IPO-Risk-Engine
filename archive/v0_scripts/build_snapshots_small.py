from __future__ import annotations

from ipo_risk_engine.features.snapshots import build_snapshot_table


def main() -> None:
    symbols = ["PLTR", "SNOW", "ABNB", "U"]
    table = build_snapshot_table(symbols)
    print(f"Built snapshot table with {table.height} rows")


if __name__ == "__main__":
    main()
