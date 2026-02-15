import polars as pl
import numpy as np
from ipo_risk_engine.state_features.spec import FeatureSpec

class FeaturePipeline:
    """
    Transforms raw lazyframe data into features using polars
    """
    def __init__(self, specs: list[FeatureSpec], bucket_ms: int):
        self.specs = specs
        self.bucket_interval = bucket_ms

    def _build_agg_expr(self, spec: FeatureSpec) -> pl.Expr:
        expr = pl.col(spec.source_col)
        if spec.filter_col is not None:
            expr = expr.filter(pl.col(spec.filter_col) == spec.filter_value)
        
        if spec.agg == "count":
            expr = expr.count()
        elif spec.agg == "sum":
            expr = expr.sum()
        elif spec.agg == "mean":
            expr = expr.mean()
        elif spec.agg == "std":
            expr = expr.std()
        
        expr = expr.alias(spec.name)

        return expr



    def transform(self, events_lf: pl.LazyFrame) -> pl.LazyFrame:
        agg_expressions = [self._build_agg_expr(spec) for spec in self.specs]
        max_period = max(self.specs, key=lambda s:s.period_ms).period_ms
        #find the max period spec then access it's period_ms field

        return (
            events_lf.group_by_dynamic(
                "ts",
                every=f"{self.bucket_interval}ms",
                period=f"{max_period}ms",
                offset=f"-{max_period}ms",
                closed="left",
            ).agg(agg_expressions)
        )
#executes aggregate expressions per the specs on the desired window
