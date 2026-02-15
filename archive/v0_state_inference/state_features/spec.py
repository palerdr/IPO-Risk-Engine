"""
Feature Definitions
"""
from dataclasses import dataclass

@dataclass
class FeatureSpec:
    name: str
    source_col: str
    agg: str #sum, mean, count, std type of aggregation
    period_ms: int
    filter_col: str | None = None
    filter_value: str | None = None

FEATURE_LIST = [
    FeatureSpec("event_count_500ms", source_col="size", agg="count", period_ms=500),
    FeatureSpec("buy_count_500ms", source_col="side", agg="count", period_ms=500, filter_col="side", filter_value="buy"),
    FeatureSpec("sell_count_500ms", source_col="side", agg="count", period_ms=500, filter_col="side", filter_value="sell"),
    FeatureSpec("size_sum_500ms", source_col="size", agg="sum", period_ms=500),
    FeatureSpec("size_mean_500ms", source_col="size", agg="mean", period_ms=500),
]