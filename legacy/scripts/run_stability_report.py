"""
Stability report: expanding temporal fold evaluation of the isotonic champion.

Tests whether the pipeline (committee → isotonic calibration → threshold tuning → policy)
produces stable, defensible results across different time periods.

Fold design (expanding train, yearly test):
  Fold 0: Train ≤ 2021, Test 2022
  Fold 1: Train ≤ 2022, Test 2023
  Fold 2: Train ≤ 2023, Test 2024
  Fold 3: Train ≤ 2024, Test 2025

Acceptance gates for "defensible v1":
  - Median FSR <= 10%
  - Worst-fold FSR <= 15%
  - Median BSS > 0
  - Median SMALL_BET >= 10%

Usage:
    python -m scripts.run_stability_report
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ipo_risk_engine.models.calibrate import (
    brier_score,
    build_oof_calibration_frame,
    _FITTERS,
    _PREDICTORS,
)
from ipo_risk_engine.models.committee import RiverCommittee, prepare_features
from ipo_risk_engine.policy.actions import (
    assign_actions,
    coverage_report,
    optimize_thresholds_constrained,
    ActionThresholds,
)

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

MAX_FSR_TUNE = 0.03
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


def evaluate_fold(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Run the full champion pipeline on one temporal fold."""
    X_train, dv_idx = prepare_features(train_df, feature_cols)
    y_train = train_df[TARGET].to_numpy()
    train_severe = train_df[SEVERE_COL].to_numpy().astype(float)

    X_test, _ = prepare_features(test_df, feature_cols)
    test_severe = test_df[SEVERE_COL].to_numpy().astype(float)
    test_adverse = test_df[ADVERSE_COL].to_numpy().astype(float)

    oof_frame = build_oof_calibration_frame(
        train_df, feature_cols, TARGET, ADVERSE_COL, n_splits=N_OOF_SPLITS,
    )
    oof_scores = oof_frame["score_oof"]
    oof_labels = oof_frame["labels"]

    sorted_train = train_df.sort("asof_date")
    sorted_severe = sorted_train[SEVERE_COL].to_numpy().astype(float)
    sorted_adverse = sorted_train[ADVERSE_COL].to_numpy().astype(float)
    block_indices = np.array_split(
        np.arange(sorted_train.height), N_OOF_SPLITS + 1,
    )
    oof_idx = np.concatenate(block_indices[1:])
    oof_severe = sorted_severe[oof_idx]
    oof_adverse = sorted_adverse[oof_idx]

    calibrator = _FITTERS[CALIBRATOR](oof_scores, oof_labels)
    oof_p_tail = _PREDICTORS[CALIBRATOR](calibrator, oof_scores)

    best_t, oof_report = optimize_thresholds_constrained(
        oof_p_tail, oof_severe, oof_adverse,
        max_false_safe_rate=MAX_FSR_TUNE,
        min_size_up_pct=MIN_SIZE_UP_PCT,
        min_small_bet_pct=MIN_SMALL_BET_PCT,
        max_fold_pct=MAX_FOLD_PCT,
    )

    committee = RiverCommittee(dollar_volume_col=dv_idx)
    committee.fit(X_train, y_train)
    test_scores = committee.predict(X_test)

    test_p_tail = _PREDICTORS[CALIBRATOR](calibrator, test_scores)
    test_actions = assign_actions(test_p_tail, best_t)

    test_report = coverage_report(test_actions, test_severe, test_adverse)
    test_bs = brier_score(test_p_tail, test_adverse)
    test_climo = brier_score(
        np.full_like(test_adverse, test_adverse.mean(), dtype=float), test_adverse,
    )
    test_bss = (1 - test_bs / test_climo) if test_climo > 0 else 0.0

    return {
        "n_train": train_df.height,
        "n_test": test_df.height,
        "n_severe_train": int(train_severe.sum()),
        "n_severe_test": int(test_severe.sum()),
        "thresholds": best_t,
        "oof_fsr": oof_report["false_safe_rate"],
        "fsr": test_report["false_safe_rate"],
        "bss": test_bss,
        "size_up_pct": test_report["size_up_pct"],
        "small_bet_pct": test_report["small_bet_pct"],
        "fold_pct": test_report["fold_pct"],
        "detection": test_report["adverse_detection_rate"],
    }


def check_acceptance_gates(fold_results: list[dict]) -> dict[str, bool]:
    """Check whether fold metrics pass defensible-v1 stability gates."""
    fsr_values = [fold["fsr"] for fold in fold_results]
    bss_values = [fold["bss"] for fold in fold_results]
    small_bet_values = [fold["small_bet_pct"] for fold in fold_results]

    median_fsr_pass = np.median(fsr_values) <= 0.10
    fsr_worst_pass = np.max(fsr_values) <= 0.15
    median_bss_pass = np.median(bss_values) > 0.0
    median_small_pass = np.median(small_bet_values) >= 0.10

    return {
        "med_fsr_pass": median_fsr_pass,
        "worst_fsr_pass": fsr_worst_pass,
        "med_bss_pass": median_bss_pass,
        "med_small_pass": median_small_pass,
        "all_pass": all([median_fsr_pass, fsr_worst_pass, median_bss_pass, median_small_pass]),
    }


def main():
    full_df = pl.read_parquet("data/dataset/snapshots_full.parquet")
    river = full_df.filter(
        pl.col("street") == "RIVER"
    ).drop_nulls(subset=[TARGET])

    feature_cols = get_feature_cols(river)

    print("=" * 65)
    print("STABILITY REPORT - Isotonic Champion")
    print("=" * 65)
    print(f"  Total RIVER rows: {river.height}")
    print(f"  Features:         {len(feature_cols)}")
    print(f"  Calibrator:       {CALIBRATOR}")
    print(f"  Tuning FSR:       <= {MAX_FSR_TUNE:.0%}")

    fold_results: list[dict] = []

    for i, fold_def in enumerate(FOLD_BOUNDARIES):
        train_end = fold_def["train_end"]
        test_year = fold_def["test_year"]

        train_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) <= train_end
        )
        test_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) == test_year
        )

        print(f"\n{'-' * 65}")
        print(f"  FOLD {i}: Train <= {train_end} ({train_df.height} rows) | "
              f"Test {test_year} ({test_df.height} rows)")
        print(f"{'-' * 65}")

        result = evaluate_fold(train_df, test_df, feature_cols)
        result["fold_id"] = i
        result["train_end"] = train_end
        result["test_year"] = test_year
        fold_results.append(result)

        t = result["thresholds"]
        print(f"    Thresholds: fold>={t.fold_min:.2f}  sbet>={t.small_bet_min:.2f}")
        print(f"    OOF FSR:    {result['oof_fsr']:.1%}")
        print(f"    Test FSR:   {result['fsr']:.1%}")
        print(f"    Test BSS:   {result['bss']:.3f}")
        print(f"    SIZE_UP:    {result['size_up_pct']:.1%}")
        print(f"    SMALL_BET:  {result['small_bet_pct']:.1%}")
        print(f"    FOLD:       {result['fold_pct']:.1%}")
        print(f"    Detection:  {result['detection']:.1%}")

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    header = f"  {'Fold':>4}  {'Test':>4}  {'FSR':>6}  {'BSS':>6}  {'SIZE_UP':>7}  {'SBET':>6}  {'FOLD':>6}  {'Det':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in fold_results:
        print(f"  {r['fold_id']:>4}  {r['test_year']:>4}  "
              f"{r['fsr']:>5.1%}  {r['bss']:>6.3f}  "
              f"{r['size_up_pct']:>6.1%}  {r['small_bet_pct']:>5.1%}  "
              f"{r['fold_pct']:>5.1%}  {r['detection']:>5.1%}")

    print(f"\n{'=' * 65}")
    print("ACCEPTANCE GATES - Defensible V1")
    print(f"{'=' * 65}")

    gates = check_acceptance_gates(fold_results)
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {gate_name:.<40s} {status}")

    manifest = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "calibrator": CALIBRATOR,
        "tuning_fsr": MAX_FSR_TUNE,
        "folds": [
            {k: (v if not isinstance(v, ActionThresholds) else
                {"fold_min": v.fold_min, "small_bet_min": v.small_bet_min})
             for k, v in r.items()}
            for r in fold_results
        ],
        "gates": gates,
    }
    manifest_path = Path("data/stability_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
