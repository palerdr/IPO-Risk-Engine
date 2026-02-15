from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class RetrievalIndex:
    feature_cols: list[str]
    label_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    street_to_matrix: dict[str, np.ndarray]
    street_to_meta: dict[str, pl.DataFrame]
    street_to_ids: dict[str, list[tuple[str, str, object]]]


def _standardize_matrix(mat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (mat - mean) / std


def fit_retrieval_index(
    snapshot_table: pl.DataFrame,
    feature_cols: Sequence[str],
    *,
    street_col: str = "street",
) -> RetrievalIndex:
    if snapshot_table.is_empty():
        raise ValueError("snapshot_table is empty")

    for col in feature_cols:
        if col not in snapshot_table.columns:
            raise ValueError(f"Missing feature column: {col}")

    label_cols = [c for c in snapshot_table.columns if c.startswith("y_")]
    if not label_cols:
        raise ValueError("No label columns found (expected y_*)")

    feature_mat = snapshot_table.select(feature_cols).to_numpy()
    mean = np.nanmean(feature_mat, axis=0)
    std = np.nanstd(feature_mat, axis=0)
    std = np.where(std == 0, 1.0, std)

    street_to_matrix: dict[str, np.ndarray] = {}
    street_to_meta: dict[str, pl.DataFrame] = {}
    street_to_ids: dict[str, list[tuple[str, str, object]]] = {}

    for street in snapshot_table.select(street_col).unique().to_series().to_list():
        street_df = snapshot_table.filter(pl.col(street_col) == street)
        mat = street_df.select(feature_cols).to_numpy()
        street_to_matrix[street] = _standardize_matrix(mat, mean, std)
        street_to_meta[street] = street_df
        ids = list(
            zip(
                street_df.get_column("symbol").to_list(),
                street_df.get_column(street_col).to_list(),
                street_df.get_column("asof").to_list(),
            )
        )
        street_to_ids[street] = ids

    return RetrievalIndex(
        feature_cols=list(feature_cols),
        label_cols=label_cols,
        mean=mean,
        std=std,
        street_to_matrix=street_to_matrix,
        street_to_meta=street_to_meta,
        street_to_ids=street_to_ids,
    )


def _x_vec_from_input(feature_cols: Sequence[str], x_vec: dict[str, float] | Sequence[float]) -> np.ndarray:
    if isinstance(x_vec, dict):
        return np.array([x_vec[c] for c in feature_cols], dtype=float)
    return np.array(list(x_vec), dtype=float)


def query(
    index: RetrievalIndex,
    street: str,
    x_vec: dict[str, float] | Sequence[float],
    *,
    k: int = 20,
    exclude_id: tuple[str, str, object] | None = None,
) -> pl.DataFrame:
    if street not in index.street_to_matrix:
        raise ValueError(f"Unknown street: {street}")

    mat = index.street_to_matrix[street]
    meta = index.street_to_meta[street]
    ids = index.street_to_ids[street]

    x = _standardize_matrix(_x_vec_from_input(index.feature_cols, x_vec), index.mean, index.std)
    distances = np.linalg.norm(mat - x, axis=1)

    if exclude_id is not None:
        mask = np.array([idx != exclude_id for idx in ids], dtype=bool)
    else:
        mask = np.ones(len(ids), dtype=bool)

    distances = distances[mask]
    meta = meta.filter(pl.Series(mask))

    if distances.size == 0:
        return pl.DataFrame()

    top_idx = np.argsort(distances)[:k]
    order_df = pl.DataFrame(
        {
            "idx": top_idx.tolist(),
            "order": list(range(len(top_idx))),
            "distance": distances[top_idx],
        }
    )
    comps = (
        meta.with_row_index("idx")
        .join(order_df, on="idx", how="inner")
        .sort("order")
        .select(["symbol", "street", "asof"] + index.label_cols + ["distance"])
    )
    return comps


def estimate_tail_risk(
    comps_df: pl.DataFrame,
    *,
    horizons: Iterable[int] = (1, 3, 5, 10),
    weighted: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    if comps_df.is_empty():
        return {}, {"K_eff": 0.0, "mean_distance": float("nan"), "nearest_distance": float("nan")}

    distances = comps_df.get_column("distance").to_numpy()
    if weighted:
        weights = 1.0 / (distances + 1e-9)
    else:
        weights = np.ones_like(distances)

    p_hat: dict[str, float] = {}
    for h in horizons:
        col = f"y_{h}"
        if col not in comps_df.columns:
            p_hat[col] = float("nan")
            continue
        vals = comps_df.get_column(col).to_numpy()
        mask = ~np.isnan(vals)
        if not mask.any():
            p_hat[col] = float("nan")
            continue
        w = weights[mask]
        p_hat[col] = float(np.sum(vals[mask] * w) / np.sum(w))

    diagnostics = {
        "K_eff": float(comps_df.height),
        "mean_distance": float(np.mean(distances)),
        "nearest_distance": float(np.min(distances)),
    }
    return p_hat, diagnostics
