"""
PREFLOP V1 ablation: 3-variant comparison against river_only baseline.

Variants (all include river_only base features):
  A (full):           + all 14 preflop V1 features
  B (no_amend_timing): + sector + s1_lead_days + exchange + s1a_count (no days_since_last_amendment)
  C (no_amendments):  + sector + s1_lead_days + exchange only (no amendment features)

Revised DoD (matched safety at MAX_FSR_TUNE=0.03):
  1. Worst-fold FSR <= 10% (hard gate)
  2. Fold-3 BSS lift >= +0.05 vs river_only
  3. Tech FSR must improve or not worsen vs river_only
  4. SMALL_BET >= 8% (relaxed from 10%, documented)

Usage:
    python -m scripts.run_preflop_v1_eval
"""
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
    optimize_thresholds_constrained,
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
N_OOF_SPLITS = 3

FOLD_BOUNDARIES = [
    {"train_end": 2021, "test_year": 2022},
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
    {"train_end": 2024, "test_year": 2025},
]

SECTOR_ONEHOT_COLS = [
    "preflop_sector_health_care",
    "preflop_sector_technology",
    "preflop_sector_industrials",
    "preflop_sector_financials",
    "preflop_sector_consumer_goods",
    "preflop_sector_consumer_services",
    "preflop_sector_energy",
    "preflop_sector_oil_gas",
    "preflop_sector_real_estate",
]
METADATA_COLS = [
    "preflop_s1_lead_days",
    "preflop_exchange_nyse",
    "preflop_exchange_nasdaq",
]
AMENDMENT_COLS = [
    "preflop_s1a_count",
    "preflop_days_since_last_amendment",
]
ALL_PREFLOP_V1_COLS = SECTOR_ONEHOT_COLS + METADATA_COLS + AMENDMENT_COLS

VARIANTS: dict[str, list[str]] = {
    "baseline":         [],  # river_only, no preflop features
    "A_full":           ALL_PREFLOP_V1_COLS,
    "B_no_amend_time":  SECTOR_ONEHOT_COLS + METADATA_COLS + ["preflop_s1a_count"],
    "C_no_amendments":  SECTOR_ONEHOT_COLS + METADATA_COLS,
}


def get_feature_cols(df: pl.DataFrame, include_cols: list[str]) -> list[str]:
    """Get feature columns for a variant."""
    exclude_preflop = set(ALL_PREFLOP_V1_COLS) - set(include_cols)
    all_feats = [
        c for c in df.columns
        if c not in EXCLUDE and c not in exclude_preflop and df[c].null_count() == 0
    ]
    return all_feats


def evaluate_fold(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Run full pipeline on one fold and return metrics."""
    X_train, dv_idx = prepare_features(train_df, feature_cols)
    y_train = train_df[TARGET].to_numpy()

    X_test, _ = prepare_features(test_df, feature_cols)
    test_severe = test_df[SEVERE_COL].to_numpy().astype(float)
    test_adverse = test_df[ADVERSE_COL].to_numpy().astype(float)

    oof_frame = build_oof_calibration_frame(
        train_df, feature_cols, TARGET, ADVERSE_COL, n_splits=N_OOF_SPLITS,
    )
    sorted_train = train_df.sort("asof_date")
    sorted_severe = sorted_train[SEVERE_COL].to_numpy().astype(float)
    sorted_adverse = sorted_train[ADVERSE_COL].to_numpy().astype(float)
    block_indices = np.array_split(np.arange(sorted_train.height), N_OOF_SPLITS + 1)
    oof_idx = np.concatenate(block_indices[1:])

    calibrator = _FITTERS[CALIBRATOR](oof_frame["score_oof"], oof_frame["labels"])
    oof_p_tail = _PREDICTORS[CALIBRATOR](calibrator, oof_frame["score_oof"])
    best_t, _ = optimize_thresholds_constrained(
        oof_p_tail, sorted_severe[oof_idx], sorted_adverse[oof_idx],
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

    from ipo_risk_engine.policy.actions import coverage_report
    report = coverage_report(test_actions, test_severe, test_adverse)
    test_bs = brier_score(test_p_tail, test_adverse)
    test_climo = brier_score(
        np.full_like(test_adverse, test_adverse.mean(), dtype=float), test_adverse,
    )
    bss = (1 - test_bs / test_climo) if test_climo > 0 else 0.0

    return {
        "fsr": report["false_safe_rate"],
        "detection": report["adverse_detection_rate"],
        "bss": bss,
        "small_bet_pct": report["small_bet_pct"],
        "size_up_pct": report["size_up_pct"],
        "fold_pct": report["fold_pct"],
        "test_sectors": test_df["sector"].to_list(),
        "test_actions": test_actions,
        "test_severe": test_severe,
        "test_adverse": test_adverse,
    }


def compute_tech_fsr(result: dict) -> float:
    """Compute FSR specifically for technology sector."""
    sectors = result["test_sectors"]
    actions = result["test_actions"]
    severe = result["test_severe"]
    tech_severe = sum(1 for s, sv in zip(sectors, severe) if s == "technology" and sv)
    tech_false_safe = sum(1 for s, a, sv in zip(sectors, actions, severe)
                         if s == "technology" and sv and a == "SIZE_UP")
    return tech_false_safe / tech_severe if tech_severe > 0 else 0.0


def main():
    full_df = pl.read_parquet("data/dataset/snapshots_full.parquet")
    river = full_df.filter(pl.col("street") == "RIVER").drop_nulls(subset=[TARGET])

    variant_cols: dict[str, list[str]] = {}
    for name, include in VARIANTS.items():
        variant_cols[name] = get_feature_cols(river, include)

    print("=" * 80)
    print("PREFLOP V1 ABLATION (3 variants + baseline)")
    print("=" * 80)
    print(f"  RIVER rows: {river.height}")
    print(f"  MAX_FSR_TUNE: {MAX_FSR_TUNE}")
    for name, cols in variant_cols.items():
        extra = len(cols) - len(variant_cols["baseline"])
        tag = f" (+{extra})" if extra else ""
        print(f"  {name:20s}: {len(cols)} features{tag}")

    results: dict[str, list[dict]] = {name: [] for name in VARIANTS}

    for i, fold_def in enumerate(FOLD_BOUNDARIES):
        train_end = fold_def["train_end"]
        test_year = fold_def["test_year"]

        train_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) <= train_end
        )
        test_df = river.filter(
            pl.col("asof_date").cast(str).str.slice(0, 4).cast(int) == test_year
        )

        print(f"\n{'=' * 80}")
        print(f"  FOLD {i}: Train <= {train_end} ({train_df.height}) | "
              f"Test {test_year} ({test_df.height})")
        print(f"{'=' * 80}")

        for name, cols in variant_cols.items():
            print(f"  {name:20s} ({len(cols)} feats)... ", end="")
            r = evaluate_fold(train_df, test_df, cols)
            r["fold_id"] = i
            r["test_year"] = test_year
            results[name].append(r)
            print(f"FSR={r['fsr']:.1%} Det={r['detection']:.1%} "
                  f"BSS={r['bss']:.3f} SB={r['small_bet_pct']:.1%}")

    print(f"\n{'=' * 80}")
    print("ABLATION COMPARISON (all variants)")
    print(f"{'=' * 80}")

    variant_names = list(VARIANTS.keys())
    header = f"  {'Fold':>4} {'Year':>4}"
    for name in variant_names:
        short = name[:8]
        header += f" | {short+'_F':>8} {short+'_D':>8} {short+'_B':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i in range(len(FOLD_BOUNDARIES)):
        row = f"  {i:>4} {results['baseline'][i]['test_year']:>4}"
        for name in variant_names:
            r = results[name][i]
            row += f" | {r['fsr']:>7.1%} {r['detection']:>7.1%} {r['bss']:>8.3f}"
        print(row)

    print("  " + "-" * (len(header) - 2))
    med_row = f"  {'Med':>4} {'':>4}"
    for name in variant_names:
        rs = results[name]
        med_row += (f" | {np.median([r['fsr'] for r in rs]):>7.1%}"
                    f" {np.median([r['detection'] for r in rs]):>7.1%}"
                    f" {np.median([r['bss'] for r in rs]):>8.3f}")
    print(med_row)

    print(f"\n{'=' * 80}")
    print("FOLD-3 DETAIL (2025 holdout)")
    print(f"{'=' * 80}")

    base_f3 = results["baseline"][3]
    base_tech_fsr = compute_tech_fsr(base_f3)
    print(f"  {'Variant':20s} {'FSR':>6} {'Det':>6} {'BSS':>6} "
          f"{'dBSS':>6} {'TechFSR':>8} {'SB%':>6} {'SU%':>6} {'FOLD%':>6}")
    print("  " + "-" * 76)

    for name in variant_names:
        r = results[name][3]
        tech_fsr = compute_tech_fsr(r)
        d_bss = r["bss"] - base_f3["bss"]
        print(f"  {name:20s} {r['fsr']:>5.1%} {r['detection']:>5.1%} {r['bss']:>6.3f} "
              f"{d_bss:>+5.3f} {tech_fsr:>7.1%} "
              f"{r['small_bet_pct']:>5.1%} {r['size_up_pct']:>5.1%} {r['fold_pct']:>5.1%}")

    print(f"\n{'=' * 80}")
    print("REVISED DEFINITION OF DONE (per variant)")
    print(f"{'=' * 80}")

    for name in variant_names:
        if name == "baseline":
            continue
        rs = results[name]
        fsrs = [r["fsr"] for r in rs]
        bsss = [r["bss"] for r in rs]
        sbets = [r["small_bet_pct"] for r in rs]
        f3 = rs[3]
        f3_bss_lift = f3["bss"] - base_f3["bss"]
        tech_fsr = compute_tech_fsr(f3)

        print(f"\n  --- {name} ---")
        checks = {
            f"Worst-fold FSR <= 10% (actual: {max(fsrs):.1%})":
                max(fsrs) <= 0.10,
            f"Fold-3 BSS lift >= +0.05 (actual: {f3_bss_lift:+.3f})":
                f3_bss_lift >= 0.05,
            f"Tech FSR not worse (base: {base_tech_fsr:.1%}, now: {tech_fsr:.1%})":
                tech_fsr <= base_tech_fsr,
            f"SMALL_BET >= 8% (median: {np.median(sbets):.1%})":
                np.median(sbets) >= 0.08,
        }
        all_pass = True
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    {check:.<62s} {status}")
        print(f"    Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
