"""
Falsification checks for V1 fold-3 drift hypothesis.

V1_LOCK.md claims fold-3's lower detection (43.5% vs 78-97% in folds 0-2) is due
to compositional shift (sector mix, liquidity), not model failure.

Three checks to test this:
  1. Reweight folds 0-2 detection to 2025 sector mix (Simpson's paradox test)
  2. Fold-3 metrics by sector (within-sector vs between-sector effect)
  3. Calibration intercept/slope by year (systematic miscalibration check)

Usage:
    python -m scripts.run_falsification_checks
"""
import numpy as np
import polars as pl

from ipo_risk_engine.models.calibrate import (
    build_oof_calibration_frame,
    _FITTERS,
    _PREDICTORS,
)
from ipo_risk_engine.models.committee import RiverCommittee, prepare_features
from ipo_risk_engine.policy.actions import (
    assign_actions,
    optimize_thresholds_constrained,
)

# --- Same constants as stability report (frozen v1) ---
TARGET = "risk_severity"
ADVERSE_COL = "adverse_20"
SEVERE_COL = "severe_30"
META_COLS = ["symbol", "street", "asof_date", "ipo_date", "sector"]
LABEL_COLS = [
    "forward_mdd_7d", "forward_mfr_7d", "forward_mdd_20d", "forward_mfr_20d",
    "risk_severity", "adverse_20", "severe_30",
]
QF_COLS = ["zero_volume_pct", "median_daily_volume", "bar_count_1d"]
EXCLUDE = META_COLS + LABEL_COLS + QF_COLS

CALIBRATOR = "isotonic"
MAX_FSR_TUNE = 0.05
MIN_SIZE_UP_PCT = 0.10
MIN_SMALL_BET_PCT = 0.12
MAX_FOLD_PCT = 0.80

FOLD_BOUNDARIES = [
    {"train_end": 2021, "test_year": 2022},
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
]
N_OOF_SPLITS = 3


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and df[c].null_count() == 0]


# ======================================================================
# Per-row fold evaluation (captures sector, p_tail, action per test row)
# ======================================================================
def evaluate_fold_detailed(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Run pipeline on a fold, return per-row test predictions."""
    X_train, dv_idx = prepare_features(train_df, feature_cols)
    y_train = train_df[TARGET].to_numpy()

    X_test, _ = prepare_features(test_df, feature_cols)
    test_severe = test_df[SEVERE_COL].to_numpy().astype(float)
    test_adverse = test_df[ADVERSE_COL].to_numpy().astype(float)

    # OOF for calibration
    oof_frame = build_oof_calibration_frame(
        train_df, feature_cols, TARGET, ADVERSE_COL, n_splits=N_OOF_SPLITS,
    )
    sorted_train = train_df.sort("asof_date")
    sorted_severe = sorted_train[SEVERE_COL].to_numpy().astype(float)
    sorted_adverse = sorted_train[ADVERSE_COL].to_numpy().astype(float)
    block_indices = np.array_split(np.arange(sorted_train.height), N_OOF_SPLITS + 1)
    oof_idx = np.concatenate(block_indices[1:])

    # Fit calibrator + tune thresholds on OOF
    calibrator = _FITTERS[CALIBRATOR](oof_frame["score_oof"], oof_frame["labels"])
    oof_p_tail = _PREDICTORS[CALIBRATOR](calibrator, oof_frame["score_oof"])
    best_t, _ = optimize_thresholds_constrained(
        oof_p_tail, sorted_severe[oof_idx], sorted_adverse[oof_idx],
        max_false_safe_rate=MAX_FSR_TUNE,
        min_size_up_pct=MIN_SIZE_UP_PCT,
        min_small_bet_pct=MIN_SMALL_BET_PCT,
        max_fold_pct=MAX_FOLD_PCT,
    )

    # Refit on all train, predict test
    committee = RiverCommittee(dollar_volume_col=dv_idx)
    committee.fit(X_train, y_train)
    test_scores = committee.predict(X_test)
    test_p_tail = _PREDICTORS[CALIBRATOR](calibrator, test_scores)
    test_actions = assign_actions(test_p_tail, best_t)

    return pl.DataFrame({
        "sector": test_df["sector"].to_list(),
        "p_tail": test_p_tail,
        "action": test_actions,
        "severe_30": test_severe,
        "adverse_20": test_adverse,
    })


# ======================================================================
# CHECK 1: Sector-reweighted detection (Simpson's paradox test)
# ======================================================================
def reweight_detection(row_df: pl.DataFrame, target_fold_id: int) -> dict[int, float]:
    """Reweight earlier folds' detection rates to the target fold's sector mix.

    TODO(human): Implement this function.

    For each source fold (fold_id != target_fold_id):
      1. Compute the target fold's sector proportions (these are the weights)
      2. Compute the source fold's per-sector detection rate
         (detection = among adverse_20==1 rows, fraction where action=="FOLD")
      3. Reweighted detection = sum over shared sectors of (weight_s * detection_s)

    Handle edge cases:
      - Only sum over sectors present in BOTH source and target
      - Renormalize weights to sum to 1 over shared sectors
      - If a sector in the source has zero adverse events, skip it

    Args:
        row_df: DataFrame with columns: fold_id, sector, action, adverse_20
        target_fold_id: The fold whose sector mix we reweight TO (e.g., 3)

    Returns:
        dict of {source_fold_id: reweighted_detection_rate}
    """

    target_df = row_df.filter(pl.col("fold_id") == target_fold_id)

    vc = target_df["sector"].value_counts()
    sector_weights = {
        row["sector"] : row["count"] / target_df.height
        for row in vc.iter_rows(named= True)
    }

    reweight = {}
    for i in range(target_fold_id):
        src_df = row_df.filter(pl.col("fold_id") == i)
        weighted_sum = 0.0
        weight_sum = 0.0
        for sector in sector_weights:
            sector_rows = src_df.filter(pl.col("sector") == sector)
            adverse_rows = sector_rows.filter(pl.col("adverse_20") == 1)

            if adverse_rows.is_empty():
                continue

            det_s = adverse_rows.filter(pl.col("action") == "FOLD").height / adverse_rows.height
            weighted_sum += sector_weights[sector] * det_s
            weight_sum += sector_weights[sector]

        reweight[i] = weighted_sum / weight_sum if weight_sum > 0 else 0.0

    return reweight
# ======================================================================
# CHECK 2: Per-sector metrics for a single fold
# ======================================================================
def check_by_sector(row_df: pl.DataFrame, fold_id: int) -> None:
    """Print per-sector metrics for a single fold."""
    fold_rows = row_df.filter(pl.col("fold_id") == fold_id)
    sectors = fold_rows["sector"].unique().sort().to_list()

    print(f"\n  {'Sector':<20s} {'N':>4} {'Sev%':>6} {'Adv%':>6} "
          f"{'Det%':>6} {'FSR%':>6} {'FOLD':>6} {'SBET':>6} {'SZUP':>6}")
    print("  " + "-" * 80)

    for s in sectors:
        s_rows = fold_rows.filter(pl.col("sector") == s)
        n = s_rows.height
        sev_rate = s_rows["severe_30"].mean()
        adv_rate = s_rows["adverse_20"].mean()

        actions = s_rows["action"].to_list()
        severe = s_rows["severe_30"].to_numpy().astype(bool)
        adverse = s_rows["adverse_20"].to_numpy().astype(bool)

        # Detection: among adverse, fraction FOLD
        if adverse.sum() > 0:
            det = sum(1 for a, m in zip(actions, adverse) if m and a == "FOLD") / adverse.sum()
        else:
            det = float("nan")

        # FSR: among severe, fraction SIZE_UP
        if severe.sum() > 0:
            fsr = sum(1 for a, m in zip(actions, severe) if m and a == "SIZE_UP") / severe.sum()
        else:
            fsr = float("nan")

        fold_pct = sum(1 for a in actions if a == "FOLD") / n
        sbet_pct = sum(1 for a in actions if a == "SMALL_BET") / n
        szup_pct = sum(1 for a in actions if a == "SIZE_UP") / n

        det_str = f"{det:5.1%}" if not np.isnan(det) else "  n/a"
        fsr_str = f"{fsr:5.1%}" if not np.isnan(fsr) else "  n/a"

        print(f"  {s:<20s} {n:>4} {sev_rate:>5.1%} {adv_rate:>5.1%} "
              f"{det_str:>6} {fsr_str:>6} "
              f"{fold_pct:>5.1%} {sbet_pct:>5.1%} {szup_pct:>5.1%}")


# ======================================================================
# CHECK 3: Calibration diagnostics by year
# ======================================================================
def check_calibration_by_year(row_df: pl.DataFrame) -> None:
    """Check calibration-in-the-large per fold.

    If calibrated: mean(p_tail) ~ mean(adverse_20).
    Ratio > 1 means model over-estimates risk (conservative).
    Ratio < 1 means model under-estimates risk (dangerous).
    OLS slope of adverse_20 ~ p_tail measures discrimination.
    """
    print(f"\n  {'Fold':>4} {'Year':>5} {'mean(p_tail)':>13} "
          f"{'mean(adv20)':>12} {'Ratio':>7} {'Slope':>7}")
    print("  " + "-" * 55)

    for fold_id in sorted(row_df["fold_id"].unique().to_list()):
        f = row_df.filter(pl.col("fold_id") == fold_id)
        year = f["test_year"][0]
        mean_p = float(f["p_tail"].mean())  # type: ignore[arg-type]
        mean_adv = float(f["adverse_20"].mean())  # type: ignore[arg-type]
        ratio = mean_p / mean_adv if mean_adv > 0 else float("nan")

        p = f["p_tail"].to_numpy()
        y = f["adverse_20"].to_numpy().astype(float)

        # OLS slope: y = a + b*p => b = cov(p,y) / var(p)
        if np.std(p) > 0:
            slope = np.cov(p, y)[0, 1] / np.var(p)
        else:
            slope = float("nan")

        print(f"  {fold_id:>4} {year:>5} {mean_p:>12.3f} "
              f"{mean_adv:>11.3f} {ratio:>6.2f} {slope:>6.3f}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    full_df = pl.read_parquet("data/dataset/snapshots_full.parquet")
    river = full_df.filter(pl.col("street") == "RIVER").drop_nulls(subset=[TARGET])
    feature_cols = get_feature_cols(river)

    print("=" * 65)
    print("FALSIFICATION CHECKS - V1 Fold-3 Drift Hypothesis")
    print("=" * 65)
    print(f"  RIVER rows: {river.height}  Features: {len(feature_cols)}")

    # ------------------------------------------------------------------
    # Run all folds, collect per-row predictions
    # ------------------------------------------------------------------
    all_rows: list[pl.DataFrame] = []
    for i, fold_def in enumerate(FOLD_BOUNDARIES):
        train_end = fold_def["train_end"]
        test_year = fold_def["test_year"]

        train_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) <= train_end
        )
        test_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) == test_year
        )

        print(f"  Running fold {i} (test {test_year}, n={test_df.height})...")
        row_df = evaluate_fold_detailed(train_df, test_df, feature_cols)
        row_df = row_df.with_columns(
            pl.lit(i).alias("fold_id"),
            pl.lit(test_year).alias("test_year"),
        )
        all_rows.append(row_df)

    row_df = pl.concat(all_rows)

    # ==================================================================
    # CHECK 1: Sector-Reweighted Detection
    # ==================================================================
    print(f"\n{'=' * 65}")
    print("CHECK 1: Sector-Reweighted Detection")
    print(f"{'=' * 65}")
    print("  Question: Does 2025's sector mix explain the detection drop?")

    reweighted = reweight_detection(row_df, target_fold_id=3)

    print(f"\n  {'Fold':>4} {'Year':>5} {'Actual Det':>11} {'Reweighted':>11}")
    print("  " + "-" * 35)
    for fold_id in sorted(row_df["fold_id"].unique().to_list()):
        f = row_df.filter(pl.col("fold_id") == fold_id)
        adv = f["adverse_20"].to_numpy().astype(bool)
        actions = f["action"].to_list()
        actual_det = (sum(1 for a, m in zip(actions, adv) if m and a == "FOLD")
                      / adv.sum()) if adv.sum() > 0 else 0.0
        year = f["test_year"][0]

        if fold_id in reweighted:
            print(f"  {fold_id:>4} {year:>5} {actual_det:>10.1%} {reweighted[fold_id]:>10.1%}")
        else:
            print(f"  {fold_id:>4} {year:>5} {actual_det:>10.1%}    (target)")

    print(f"\n  Interpretation: if reweighted values drop toward fold 3's actual,")
    print(f"  the detection decline is explained by sector mix shift.")

    # ==================================================================
    # CHECK 2: Fold-3 Per-Sector Metrics
    # ==================================================================
    print(f"\n{'=' * 65}")
    print("CHECK 2: Fold-3 Per-Sector Metrics")
    print(f"{'=' * 65}")
    print("  Question: Is detection low across all sectors, or driven by financials?")
    check_by_sector(row_df, fold_id=3)

    # Also show fold-2 for comparison
    print(f"\n  --- For comparison: Fold-2 (2024) ---")
    check_by_sector(row_df, fold_id=2)

    # ==================================================================
    # CHECK 3: Calibration by Year
    # ==================================================================
    print(f"\n{'=' * 65}")
    print("CHECK 3: Calibration Diagnostics by Year")
    print(f"{'=' * 65}")
    print("  Question: Is the model systematically over-risking 2025?")
    print("  Ratio > 1 = conservative (over-estimates risk)")
    print("  Ratio < 1 = aggressive (under-estimates risk)")
    check_calibration_by_year(row_df)


if __name__ == "__main__":
    main()
