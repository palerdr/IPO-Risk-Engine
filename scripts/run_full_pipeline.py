"""
Full pipeline: OOF committee -> hardened calibration -> policy -> held-out test.

Phase 3.5: True OOF committee scores, hardened calibrator selection with bootstrap CI
Phase 4:   Policy layer with threshold optimization (false-safe constraint)
Phase 5:   Held-out test evaluation, leakage tests, run manifest

Usage:
    python -m scripts.run_full_pipeline
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ipo_risk_engine.models.calibrate import (
    brier_score,
    build_oof_calibration_frame,
    select_calibrator_hardened,
    predict_p_tail,
    select_calibrator_oof,
)
from ipo_risk_engine.models.committee import RiverCommittee, prepare_features
from ipo_risk_engine.policy.actions import (
    assign_actions,
    assign_actions_by_rank,
    coverage_report,
    optimize_thresholds_constrained,
    should_calibrate,
    summarize_actions,
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
SEED = 42


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and df[c].null_count() == 0]


def hash_parquet(path: str) -> str:
    """SHA256 of parquet file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    train_df = pl.read_parquet("data/dataset/train.parquet")
    val_df = pl.read_parquet("data/dataset/val.parquet")
    test_df = pl.read_parquet("data/dataset/test.parquet")

    river_train = train_df.filter(pl.col("street") == "RIVER").drop_nulls(subset=[TARGET])
    river_val = val_df.filter(pl.col("street") == "RIVER").drop_nulls(subset=[TARGET])
    river_test = test_df.filter(pl.col("street") == "RIVER").drop_nulls(subset=[TARGET])

    river_trainval = pl.concat([river_train, river_val])
    feature_cols = get_feature_cols(river_trainval)

    trainval_severe = river_trainval[SEVERE_COL].to_numpy().astype(float)
    trainval_adverse = river_trainval[ADVERSE_COL].to_numpy().astype(float)
    test_adverse = river_test[ADVERSE_COL].to_numpy().astype(float)
    test_severe = river_test[SEVERE_COL].to_numpy().astype(float)

    print("=" * 60)
    print("PHASE 3.5: OOF Committee Scores + Calibration")
    print("=" * 60)
    print(f"  Train+Val: {river_trainval.height} rows, Test: {river_test.height} rows")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Severe events (train+val): {int(trainval_severe.sum())}")

    oof_frame = build_oof_calibration_frame(
        river_trainval, feature_cols, TARGET, ADVERSE_COL, n_splits=3
    )
    print(f"  OOF samples: {len(oof_frame['score_oof'])}")
    print(f"  OOF adverse rate: {oof_frame['labels'].mean():.1%}")
    print(f"  OOF score range: [{oof_frame['score_oof'].min():.4f}, "
          f"{oof_frame['score_oof'].max():.4f}]")

    use_calibration = should_calibrate(trainval_severe)
    print(f"\n  Calibration gate: {'PASS' if use_calibration else 'FAIL'} "
          f"({int(trainval_severe.sum())} severe events, need 50)")

    cal_result = None
    if use_calibration:
        print()
        cal_result = select_calibrator_hardened(oof_frame, n_folds=3, seed=SEED)

    X_trainval, dv_idx = prepare_features(river_trainval, feature_cols)
    y_trainval = river_trainval[TARGET].to_numpy()

    final_committee = RiverCommittee(dollar_volume_col=dv_idx)
    final_committee.fit(X_trainval, y_trainval)
    trainval_scores = final_committee.predict(X_trainval)

    print()
    print("=" * 60)
    print("PHASE 4: Policy Layer")
    print("=" * 60)

    if use_calibration and cal_result is not None:
        print("  Mode: CALIBRATED probability thresholds")
        trainval_p_tail = predict_p_tail(cal_result, trainval_scores)
        print(f"  p_tail range: [{trainval_p_tail.min():.3f}, {trainval_p_tail.max():.3f}]")
        print(f"  p_tail unique: {len(np.unique(np.round(trainval_p_tail, 4)))}")

        best_thresholds, opt_report = optimize_thresholds_constrained(
            trainval_p_tail, trainval_severe, trainval_adverse,
            max_false_safe_rate=0.10, min_size_up_pct=0.10, max_fold_pct=0.80,
        )
        trainval_actions = assign_actions(trainval_p_tail, best_thresholds)
        print(f"\n  Optimized thresholds: fold>={best_thresholds.fold_min:.2f}  "
              f"sbet>={best_thresholds.small_bet_min:.2f}")
    else:
        print("  Mode: RANKING policy (calibration gate failed)")
        trainval_actions = assign_actions_by_rank(trainval_scores)
        opt_report = coverage_report(trainval_actions, trainval_severe, trainval_adverse)
        best_thresholds = None

    print("  Train+Val policy mix:")
    for k, v in opt_report.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

    print()
    print("=" * 60)
    print("PHASE 5: Held-Out Test (ONE LOOK)")
    print("=" * 60)

    X_test, _ = prepare_features(river_test, feature_cols)
    test_scores = final_committee.predict(X_test)

    if use_calibration and cal_result is not None:
        test_p_tail = predict_p_tail(cal_result, test_scores)
        test_brier = brier_score(test_p_tail, test_adverse)
        test_base_rate = test_adverse.mean()
        test_brier_climo = brier_score(
            np.full_like(test_adverse, test_base_rate, dtype=float), test_adverse
        )
        test_bss = (1 - test_brier / test_brier_climo) if test_brier_climo > 0 else 0.0

        print(f"  Test N={len(test_adverse)}")
        print(f"  Test adverse rate: {test_base_rate:.1%}")
        print(f"  Test severe rate:  {test_severe.mean():.1%}")
        print(f"  Test Brier:        {test_brier:.4f}")
        print(f"  Test Brier climo:  {test_brier_climo:.4f}")
        print(f"  Test BSS:          {test_bss:.3f}")
        test_actions = assign_actions(test_p_tail, best_thresholds)
    else:
        test_bss = None
        print(f"  Test N={len(test_adverse)}")
        print(f"  Test adverse rate: {test_adverse.mean():.1%}")
        print(f"  Test severe rate:  {test_severe.mean():.1%}")
        print("  (BSS not computed - ranking mode, no calibration)")
        test_actions = assign_actions_by_rank(test_scores)

    test_report = coverage_report(test_actions, test_severe, test_adverse)
    print(f"\n  Test policy mix:")
    for k, v in test_report.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

    print()
    print("-" * 40)
    print("Shuffle-Leakage Test")
    print("-" * 40)
    rng = np.random.default_rng(SEED)
    n_shuffles = 100
    shuffle_bss = []
    for _ in range(n_shuffles):
        shuffled_labels = rng.permutation(oof_frame["labels"])
        shuf_result = select_calibrator_oof(oof_frame["score_oof"], shuffled_labels, n_folds=3)
        best_shuf_bss = max(shuf_result.bss_by_method.values())
        shuffle_bss.append(best_shuf_bss)

    shuffle_bss_arr = np.array(shuffle_bss)
    real_bss = max(cal_result.bss_by_method.values()) if cal_result else 0.0
    p_value = float((shuffle_bss_arr >= real_bss).mean())

    print(f"  Real best BSS:     {real_bss:.3f}")
    print(f"  Shuffle BSS mean:  {shuffle_bss_arr.mean():.3f}")
    print(f"  Shuffle BSS 95th:  {np.percentile(shuffle_bss_arr, 95):.3f}")
    print(f"  p-value:           {p_value:.3f}")
    print(f"  Leakage detected:  {'YES' if p_value > 0.5 else 'No'}")

    print()
    print("-" * 40)
    print("Future-Shift Sentinel Test")
    print("-" * 40)
    shifted_labels = np.roll(oof_frame["labels"], shift=5)
    shift_result = select_calibrator_oof(oof_frame["score_oof"], shifted_labels, n_folds=3)
    shift_bss = max(shift_result.bss_by_method.values())
    print(f"  Shifted BSS:       {shift_bss:.3f}")
    print(f"  Real BSS:          {real_bss:.3f}")
    print(f"  Shift destroys signal: {'Yes (good)' if shift_bss < real_bss else 'NO - investigate'}")

    manifest = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "dataset_hashes": {
            "train": hash_parquet("data/dataset/train.parquet"),
            "val": hash_parquet("data/dataset/val.parquet"),
            "test": hash_parquet("data/dataset/test.parquet"),
        },
        "split_sizes": {
            "river_train": river_train.height,
            "river_val": river_val.height,
            "river_test": river_test.height,
        },
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "calibration_mode": "calibrated" if use_calibration else "ranking",
        "calibrator": cal_result.selected if cal_result else None,
        "oof_bss": cal_result.bss_by_method if cal_result else None,
        "test_bss": test_bss,
        "thresholds": {
            "fold_min": best_thresholds.fold_min,
            "small_bet_min": best_thresholds.small_bet_min,
        } if best_thresholds else None,
        "test_false_safe_rate": test_report["false_safe_rate"],
        "test_adverse_detection": test_report["adverse_detection_rate"],
        "shuffle_p_value": p_value,
        "shift_bss": shift_bss,
        "test_action_mix": summarize_actions(test_actions),
    }

    manifest_path = Path("data/run_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Mode:              {'calibrated' if use_calibration else 'ranking'}")
    if cal_result:
        print(f"  Calibrator:        {cal_result.selected}")
        print(f"  OOF BSS:           {cal_result.bss_by_method[cal_result.selected]:.3f}")
    if test_bss is not None:
        print(f"  Test BSS:          {test_bss:.3f}")
    if best_thresholds:
        print(f"  Thresholds:        fold>={best_thresholds.fold_min:.2f}  "
              f"sbet>={best_thresholds.small_bet_min:.2f}")
    print(f"  Test false-safe:   {test_report['false_safe_rate']:.1%}")
    print(f"  Test detection:    {test_report['adverse_detection_rate']:.1%}")
    print(f"  Shuffle p-value:   {p_value:.3f}")
    print(f"  Shift sentinel:    {'PASS' if shift_bss < real_bss else 'FAIL'}")
    print(f"  Action mix:        {summarize_actions(test_actions)}")
    print(f"  Manifest:          {manifest_path}")


if __name__ == "__main__":
    main()
