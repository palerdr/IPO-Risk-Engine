"""Fix: recompute preflop_sector_ipo_heat_90d and preflop_sector_return_20d
that were accidentally dropped by augment_preflop_v1.py."""
import polars as pl
from ipo_risk_engine.features.preflop_features import _sector_heat, _pre_ipo_market_stats
from ipo_risk_engine.features.regime_features import get_sector_etf
from ipo_risk_engine.data.store import read_parquet, raw_bars_path

univ = pl.read_parquet("data/ipo_universe_alpaca_v1.parquet")
snaps = pl.read_parquet("data/dataset/snapshots_full.parquet")
print(f"Snapshots: {snaps.shape}")

sic_map = {}
for row in univ.iter_rows(named=True):
    if row["sic"]:
        sic_map[row["symbol"]] = row["sic"]

# Recompute per symbol
symbols = snaps["symbol"].unique().to_list()
heat_map: dict[str, float] = {}
ret_map: dict[str, float] = {}

for sym in symbols:
    sym_rows = snaps.filter(pl.col("symbol") == sym)
    ipo_date = sym_rows["ipo_date"][0]
    sector = sym_rows["sector"][0]
    sic = sic_map.get(sym)

    heat_map[sym] = _sector_heat(ipo_date, sic, univ)

    etf_sym = get_sector_etf(sector)
    sector_ret = 0.0
    if etf_sym:
        etf_path = raw_bars_path(etf_sym, "1d")
        if etf_path.exists():
            etf_bars = read_parquet(etf_path)
            sector_ret, _ = _pre_ipo_market_stats(etf_bars, ipo_date, n_days=20)
    ret_map[sym] = sector_ret

# Add columns
heat_series = snaps["symbol"].map_elements(
    lambda s: heat_map.get(s, 0.0), return_dtype=pl.Float64
)
ret_series = snaps["symbol"].map_elements(
    lambda s: ret_map.get(s, 0.0), return_dtype=pl.Float64
)
snaps = snaps.with_columns([
    heat_series.alias("preflop_sector_ipo_heat_90d"),
    ret_series.alias("preflop_sector_return_20d"),
])

print(f"Shape: {snaps.shape}")
print(f"Heat: nulls={snaps['preflop_sector_ipo_heat_90d'].null_count()}, "
      f"mean={snaps['preflop_sector_ipo_heat_90d'].mean():.2f}")
print(f"Ret:  nulls={snaps['preflop_sector_return_20d'].null_count()}, "
      f"mean={snaps['preflop_sector_return_20d'].mean():.4f}")

snaps.write_parquet("data/dataset/snapshots_full.parquet")
print("Saved")
