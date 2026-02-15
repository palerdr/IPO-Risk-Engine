import polars as pl
"""
Joins features and labels into a supervised dataset
"""
def build_supervised(features_lf: pl.LazyFrame, labels_lf:pl.LazyFrame) -> pl.LazyFrame:
    #inner join on timestamps
    return features_lf.join(labels_lf, on="ts", how="inner")
