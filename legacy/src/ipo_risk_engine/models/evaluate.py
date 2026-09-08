import numpy as np
import polars as pl


def evaluate_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    return {"mae": float(mae), "mse": float(mse)}


def score_subset(df: pl.DataFrame, pred_col: str, label_col: str) -> dict[str, float]:
    preds = df[pred_col].to_numpy()
    labels = df[label_col].to_numpy()
    return evaluate_predictions(preds, labels)


def evaluate_by_group(
    df: pl.DataFrame,
    pred_col: str,
    label_col: str,
    group_col: str,
) -> dict[str, dict[str, float]]:
    """Score predictions broken down by a grouping column."""
    results: dict[str, dict[str, float]] = {"_overall": score_subset(df, pred_col, label_col)}
    for group_val in df[group_col].unique().sort().to_list():
        subset = df.filter(pl.col(group_col) == group_val)
        results[str(group_val)] = score_subset(subset, pred_col, label_col)
    return results
