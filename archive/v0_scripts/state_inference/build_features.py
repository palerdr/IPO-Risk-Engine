"""
Loads events.parquet and runs the FeaturePipeline.transform() to compute atomic features
Computes derived features and then saves features.parquet
"""
from pathlib import Path

import polars as pl
from ipo_risk_engine.state_features.pipeline import FeaturePipeline
from ipo_risk_engine.state_features.spec import FEATURE_LIST
from ipo_risk_engine.state_data.build_supervised import build_supervised

def main():
    events_lf = pl.scan_parquet("data/raw/events.parquet")
    labels_lf = pl.scan_parquet("data/raw/labels.parquet")

    pipeline = FeaturePipeline(FEATURE_LIST, bucket_ms=100)
    features_lf = pipeline.transform(events_lf)

    #compute any derived features
    features_lf = features_lf.with_columns([
    pl.when(pl.col("buy_count_500ms") + pl.col("sell_count_500ms") == 0)
    .then(0.0)
    .otherwise(
        (pl.col("buy_count_500ms").cast(pl.Float64) - pl.col("sell_count_500ms").cast(pl.Float64))
        / (pl.col("buy_count_500ms") + pl.col("sell_count_500ms"))
    )
    .alias("buy_imbalance_500ms"),
    ])

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    

    supervised_lf = build_supervised(features_lf, labels_lf)
    supervised_df = supervised_lf.collect()
    supervised_df.write_parquet("data/processed/supervised.parquet")

    print(f"SUPERVISED shape: {supervised_df.height} rows, {supervised_df.width} columns")



if __name__ == "__main__":
    main()