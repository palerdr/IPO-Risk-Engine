"""
Street-stratified baseline evaluation.

Street roles:
  FLOP/TURN: constant model (mean risk_severity) — too early for real signal
  RIVER:     Ridge(alpha=1000) + KNN(k=7) — main modeling street

Reports per street:
  - MAE, MSE, Pearson correlation for risk_severity regression
  - Brier score and Brier Skill Score (vs climatology) for tail_event classification

Usage:
    python -m scripts.run_baselines
"""
import numpy as np
import polars as pl

from ipo_risk_engine.models.baseline import BaselineModel, BaseRidge, FiveKNN
from ipo_risk_engine.models.evaluate import evaluate_predictions
from ipo_risk_engine.models.calibrate import brier_score

TARGET = "risk_severity"
ADVERSE_COL = "adverse_event_15"
SEVERE_COL = "severe_tail_20"
META_COLS = ["symbol", "street", "asof_date", "sector"]
LABEL_COLS = [
    "forward_mdd_7d", "forward_mfr_7d", "forward_mdd_20d", "forward_mfr_20d",
    "risk_severity", "adverse_event_15", "severe_tail_20",
]
QF_COLS = ["zero_volume_pct", "median_5m_volume", "bar_count_5m"]
EXCLUDE = META_COLS + LABEL_COLS + QF_COLS


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature columns that are non-null for this street subset."""
    return [
        c for c in df.columns
        if c not in EXCLUDE and df[c].null_count() == 0
    ]


def log_transform_dollar_volume(X: np.ndarray, col_idx: int) -> np.ndarray:
    """Apply log1p to dollar_volume_mean column in-place."""
    X = X.copy()
    X[:, col_idx] = np.log1p(X[:, col_idx])
    return X


def report_street(
    street: str,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
) -> None:
    """Evaluate baselines for one street."""
    print(f"\n{'='*60}")
    print(f"STREET: {street}")
    print(f"{'='*60}")

    feature_cols = get_feature_cols(train_df)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Train: {train_df.height} rows | Test: {test_df.height} rows")

    train_y = train_df[TARGET].to_numpy()
    test_y = test_df[TARGET].to_numpy()
    train_adverse = train_df[ADVERSE_COL].to_numpy().astype(float)
    test_adverse = test_df[ADVERSE_COL].to_numpy().astype(float)

    # Climatology baseline: always predict the training base rate
    base_rate = train_adverse.mean()
    climo_probs = np.full_like(test_adverse, base_rate)
    brier_climo = brier_score(climo_probs, test_adverse)
    print(f"  Train adverse_15 rate: {base_rate:.1%}")
    print(f"  Test  adverse_15 rate: {test_adverse.mean():.1%}")
    print(f"  Brier (climatology):   {brier_climo:.4f}")

    # --- Constant model (mean predictor) ---
    const = BaselineModel().fit(None, train_y)
    const_preds = const.predict(test_y)  # shape doesn't matter, returns constant
    const_metrics = evaluate_predictions(const_preds, test_y)
    print(f"\n  Constant:  MAE={const_metrics['mae']:.4f}  "
          f"MSE={const_metrics['mse']:.6f}")

    if street in ("FLOP", "TURN"):
        print("  -> Prior-only street. No regression models fitted.")
        return

    # --- RIVER: fit Ridge + KNN ---
    train_X = train_df.select(feature_cols).to_numpy().astype(float)
    test_X = test_df.select(feature_cols).to_numpy().astype(float)

    # Log-transform dollar_volume_mean if present
    if "dollar_volume_mean" in feature_cols:
        dv_idx = feature_cols.index("dollar_volume_mean")
        train_X = log_transform_dollar_volume(train_X, dv_idx)
        test_X = log_transform_dollar_volume(test_X, dv_idx)

    # Ridge(alpha=1000)
    ridge = BaseRidge(alpha=1000)
    ridge.fit(train_X, train_y)
    ridge_preds = ridge.predict(test_X)
    ridge_metrics = evaluate_predictions(ridge_preds, test_y)

    # KNN(k=7)
    knn = FiveKNN(n_neighbors=7)
    knn.fit(train_X, train_y)
    knn_preds = knn.predict(test_X)
    knn_metrics = evaluate_predictions(knn_preds, test_y)

    print(f"  Ridge(1000): MAE={ridge_metrics['mae']:.4f}  "
          f"MSE={ridge_metrics['mse']:.6f}")
    print(f"  KNN(7):      MAE={knn_metrics['mae']:.4f}  "
          f"MSE={knn_metrics['mse']:.6f}")

    # Brier scores using risk_severity as uncalibrated probability proxy
    # (Just to establish a reference — real calibration comes in Phase 3)
    # Threshold risk_severity predictions to get crude tail probabilities
    for name, preds in [("Ridge", ridge_preds), ("KNN", knn_preds)]:
        crude_probs = np.clip(preds / 0.25, 0, 1)  # linear scaling: severity/threshold
        bs = brier_score(crude_probs, test_adverse)
        bss = 1 - bs / brier_climo if brier_climo > 0 else 0.0
        print(f"  {name} crude Brier={bs:.4f}  BSS={bss:.3f}")


def main():
    train_df = pl.read_parquet("data/dataset/train.parquet")
    test_df = pl.read_parquet("data/dataset/test.parquet")

    for street in ["FLOP", "TURN", "RIVER"]:
        st_train = train_df.filter(pl.col("street") == street).drop_nulls(subset=[TARGET])
        st_test = test_df.filter(pl.col("street") == street).drop_nulls(subset=[TARGET])

        if st_test.height == 0:
            print(f"\n{street}: no test data, skipping.")
            continue

        report_street(street, st_train, st_test)

    print(f"\n{'='*60}")
    print("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
