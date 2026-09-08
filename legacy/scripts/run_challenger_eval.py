"""
Challenger evaluation: compare calibration strategies on OOF train+val data.

Challengers:
  1. Platt (sigmoid / logistic regression on 1D scores)
  2. Isotonic regression — more granular
  3. Binned (n_bins=4) — current default
  4. Raw ranking — no calibration, quantile-based policy

Structural gates:
  Pre-tuning:   unique(p_tail) >= 8      (ensures policy granularity)
  Post-tuning:  SMALL_BET in [10%, 40%]  (prevents policy collapse)

Tuning: FSR <= 5% on OOF data (stricter than 10% to leave shift margin).

Usage:
    python -m scripts.run_challenger_eval
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from ipo_risk_engine.dataset.assemble import temporal_train_val_test_split
from ipo_risk_engine.models.calibrate import (
    brier_score,
    build_oof_calibration_frame,
    calibrate_oof_all_methods,
    _FITTERS,
    _PREDICTORS,
)
from ipo_risk_engine.models.committee import RiverCommittee, prepare_features
from ipo_risk_engine.policy.actions import (
    assign_actions,
    assign_actions_by_rank,
    coverage_report,
    optimize_thresholds_constrained,
    should_calibrate,
    feasibility_report,
    summarize_actions,
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
SEED = 42
N_OOF_SPLITS = 3
N_CAL_FOLDS = 3

MIN_UNIQUE_P_TAIL = 8
MIN_SMALL_BET_PCT = 0.10
MAX_SMALL_BET_PCT = 0.40

MAX_FSR_TUNE = 0.05
MIN_SIZE_UP_PCT = 0.10
MAX_FOLD_PCT = 0.80


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE and df[c].null_count() == 0]


def evaluate_calibrated_challenger(
    name: str,
    oof_preds: np.ndarray,
    oof_severe: np.ndarray,
    oof_adverse: np.ndarray,
    brier_climo: float,
    oof_labels: np.ndarray,
) -> dict:
    """Evaluate one calibrated challenger."""
    bs = brier_score(oof_preds, oof_labels)
    bss = (1 - bs / brier_climo) if brier_climo > 0 else 0.0
    n_unique = len(np.unique(np.round(oof_preds, 4)))

    granularity_ok = n_unique >= MIN_UNIQUE_P_TAIL

    best_t, report = optimize_thresholds_constrained(
        oof_preds, oof_severe, oof_adverse,
        max_false_safe_rate=MAX_FSR_TUNE,
        min_size_up_pct=MIN_SIZE_UP_PCT,
        max_fold_pct=MAX_FOLD_PCT,
    )

    sbet_pct = report["small_bet_pct"]
    coverage_ok = MIN_SMALL_BET_PCT <= sbet_pct <= MAX_SMALL_BET_PCT
    gate_passed = granularity_ok and coverage_ok

    return {
        "name": name,
        "bss": bss,
        "brier": bs,
        "n_unique": n_unique,
        "granularity_ok": granularity_ok,
        "coverage_ok": coverage_ok,
        "gate_passed": gate_passed,
        "thresholds": best_t,
        "fsr": report["false_safe_rate"],
        "size_up_pct": report["size_up_pct"],
        "small_bet_pct": sbet_pct,
        "fold_pct": report["fold_pct"],
        "detection": report["adverse_detection_rate"],
    }


def evaluate_ranking_challenger(
    oof_scores: np.ndarray,
    oof_severe: np.ndarray,
    oof_adverse: np.ndarray,
) -> dict:
    """Grid-search ranking quantile thresholds under FSR constraint."""
    best_q = (0.75, 0.50)
    best_report = coverage_report(
        assign_actions_by_rank(oof_scores, 0.75, 0.50), oof_severe, oof_adverse
    )
    best_score = -1.0  # will be replaced on first feasible hit

    for fold_q in np.arange(0.50, 0.96, 0.05):
        for sbet_q in np.arange(0.20, fold_q, 0.05):
            actions = assign_actions_by_rank(oof_scores, fold_q, sbet_q)
            report = coverage_report(actions, oof_severe, oof_adverse)
            if report["false_safe_rate"] > MAX_FSR_TUNE:
                continue
            if report["size_up_pct"] < MIN_SIZE_UP_PCT:
                continue
            if report["fold_pct"] > MAX_FOLD_PCT:
                continue
            if report["size_up_pct"] > best_score:
                best_score = report["size_up_pct"]
                best_q = (float(fold_q), float(sbet_q))
                best_report = report

    sbet_pct = best_report["small_bet_pct"]
    coverage_ok = MIN_SMALL_BET_PCT <= sbet_pct <= MAX_SMALL_BET_PCT

    return {
        "name": "ranking",
        "bss": None,
        "brier": None,
        "n_unique": None,
        "granularity_ok": True,  # N/A for ranking, pass by default
        "coverage_ok": coverage_ok,
        "gate_passed": coverage_ok,  # only coverage gate applies
        "quantiles": best_q,
        "fsr": best_report["false_safe_rate"],
        "size_up_pct": best_report["size_up_pct"],
        "small_bet_pct": sbet_pct,
        "fold_pct": best_report["fold_pct"],
        "detection": best_report["adverse_detection_rate"],
    }


def select_champion(challengers: dict[str, dict]) -> str:
    """Select the best challenger by: safety → coverage → BSS.

    Args:
        challengers: dict of name → evaluation result dict.
            Each dict has keys: gate_passed, fsr, size_up_pct, bss (or None).

    Returns:
        Name of the champion challenger.
    """

    passed = {n : c for n , c in challengers.items() if c["gate_passed"]}
    failed = {n : c for n , c in challengers.items() if not c["gate_passed"]}
    
    pool = passed if passed else failed
    if pool == failed : raise RuntimeError("No challengers passed gates")
    
    def sort_key(name):
        c = pool[name]
        fsr = c["fsr"]
        sizeup = -c["size_up_pct"]
        if c["bss"] is not None:
            bss = -c["bss"]
        else:
            bss = float('inf')
        return (fsr, sizeup, bss)
    
    champion = min(pool, key = sort_key)

    return champion

def print_challenger(c: dict) -> None:
    """Pretty-print a single challenger result."""
    print(f"\n  --- {c['name'].upper()} ---")
    if c["bss"] is not None:
        print(f"    BSS:        {c['bss']:.3f}")
        print(f"    Unique:     {c['n_unique']}")
        gate_str = "PASS" if c["granularity_ok"] else "FAIL"
        print(f"    Gran. gate: {gate_str} (need >= {MIN_UNIQUE_P_TAIL})")
    else:
        print(f"    BSS:        N/A (ranking)")
        print(f"    Quantiles:  fold={c['quantiles'][0]:.2f}  sbet={c['quantiles'][1]:.2f}")
    if "thresholds" in c and c["thresholds"] is not None:
        t = c["thresholds"]
        print(f"    Thresholds: fold>={t.fold_min:.2f}  sbet>={t.small_bet_min:.2f}")
    print(f"    FSR:        {c['fsr']:.1%}")
    print(f"    SIZE_UP:    {c['size_up_pct']:.1%}")
    print(f"    SMALL_BET:  {c['small_bet_pct']:.1%}")
    print(f"    FOLD:       {c['fold_pct']:.1%}")
    print(f"    Detection:  {c['detection']:.1%}")
    cov_str = "PASS" if c["coverage_ok"] else "FAIL"
    print(f"    Cov. gate:  {cov_str} (SBET in [{MIN_SMALL_BET_PCT:.0%}, {MAX_SMALL_BET_PCT:.0%}])")
    print(f"    ALL GATES:  {'PASS' if c['gate_passed'] else 'FAIL'}")


def main():
    full_df = pl.read_parquet("data/dataset/snapshots_full.parquet")

    trainval_df, holdout_df, _ = temporal_train_val_test_split(
        full_df, train_frac=0.70, val_frac=0.30,
    )

    river_trainval = trainval_df.filter(
        pl.col("street") == "RIVER"
    ).drop_nulls(subset=[TARGET])
    river_holdout = holdout_df.filter(
        pl.col("street") == "RIVER"
    ).drop_nulls(subset=[TARGET])

    feature_cols = get_feature_cols(river_trainval)

    trainval_severe = river_trainval[SEVERE_COL].to_numpy().astype(float)
    holdout_severe = river_holdout[SEVERE_COL].to_numpy().astype(float)
    holdout_adverse = river_holdout[ADVERSE_COL].to_numpy().astype(float)

    print("=" * 60)
    print("CHALLENGER EVALUATION")
    print("=" * 60)
    print(f"  Train+Val: {river_trainval.height} RIVER rows")
    print(f"  Holdout:   {river_holdout.height} RIVER rows")
    print(f"  Features:  {len(feature_cols)}")
    print(f"  Severe (train+val): {int(trainval_severe.sum())} "
          f"({trainval_severe.mean():.1%})")
    print(f"  Severe (holdout):   {int(holdout_severe.sum())} "
          f"({holdout_severe.mean():.1%})")

    feas = feasibility_report(trainval_severe, max_false_safe_rate=MAX_FSR_TUNE)
    print(f"\n  Feasibility (FSR <= {MAX_FSR_TUNE:.0%}):")
    print(f"    p_severe:           {feas['p_severe']:.3f}")
    print(f"    max_feasible_sizeup: {feas['max_feasible_sizeup']:.3f}")

    use_cal = should_calibrate(trainval_severe)
    print(f"\n  Calibration gate: {'PASS' if use_cal else 'FAIL'} "
          f"({int(trainval_severe.sum())} severe, need 50)")

    print("\n" + "=" * 60)
    print("PHASE A: OOF Committee Scores")
    print("=" * 60)

    oof_frame = build_oof_calibration_frame(
        river_trainval, feature_cols, TARGET, ADVERSE_COL, n_splits=N_OOF_SPLITS,
    )
    oof_scores = oof_frame["score_oof"]
    oof_adverse_labels = oof_frame["labels"]

    sorted_trainval = river_trainval.sort("asof_date")
    sorted_severe = sorted_trainval[SEVERE_COL].to_numpy().astype(float)
    sorted_adverse = sorted_trainval[ADVERSE_COL].to_numpy().astype(float)
    committee_blocks = np.array_split(
        np.arange(sorted_trainval.height), N_OOF_SPLITS + 1,
    )
    committee_oof_idx = np.concatenate(committee_blocks[1:])
    oof_severe = sorted_severe[committee_oof_idx]
    oof_adverse = sorted_adverse[committee_oof_idx]

    print(f"  OOF samples:     {len(oof_scores)}")
    print(f"  OOF severe rate: {oof_severe.mean():.1%}")
    print(f"  OOF score range: [{oof_scores.min():.4f}, {oof_scores.max():.4f}]")

    print("\n" + "=" * 60)
    print("PHASE B: Calibrated Challengers (nested OOF)")
    print("=" * 60)

    challengers: dict[str, dict] = {}

    cal_oof_idx = np.arange(len(oof_scores))
    cal_severe = oof_severe
    cal_adverse = oof_adverse

    if use_cal:
        cal_preds, cal_oof_idx, cal_oof_labels, brier_climo = (
            calibrate_oof_all_methods(oof_scores, oof_adverse_labels, n_folds=N_CAL_FOLDS)
        )
        cal_severe = oof_severe[cal_oof_idx]
        cal_adverse = oof_adverse[cal_oof_idx]

        print(f"  Calibrator OOF samples: {len(cal_oof_labels)}")
        print(f"  Brier climatology:      {brier_climo:.4f}")

        for method_name, preds in cal_preds.items():
            result = evaluate_calibrated_challenger(
                method_name, preds, cal_severe, cal_adverse,
                brier_climo, cal_oof_labels,
            )
            challengers[method_name] = result
            print_challenger(result)
    else:
        print("  Calibration gate FAILED — skipping calibrated challengers")

    print("\n" + "=" * 60)
    print("PHASE C: Ranking Challenger")
    print("=" * 60)

    if use_cal:
        rank_scores = oof_scores[cal_oof_idx]
        rank_severe = cal_severe
        rank_adverse = cal_adverse
    else:
        rank_scores = oof_scores
        rank_severe = oof_severe
        rank_adverse = oof_adverse

    rank_result = evaluate_ranking_challenger(rank_scores, rank_severe, rank_adverse)
    challengers["ranking"] = rank_result
    print_challenger(rank_result)

    print("\n" + "=" * 60)
    print("PHASE D: Champion Selection")
    print("=" * 60)

    passing = [n for n, c in challengers.items() if c["gate_passed"]]
    failing = [n for n, c in challengers.items() if not c["gate_passed"]]
    print(f"  Gates passed: {passing}")
    print(f"  Gates failed: {failing}")

    champion_name = select_champion(challengers)
    champion = challengers[champion_name]
    print(f"\n  CHAMPION: {champion_name.upper()}")

    # ================================================================
    # PHASE E: REFIT + HOLDOUT EVALUATION (ONE LOOK)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE E: Holdout Evaluation (ONE LOOK)")
    print("=" * 60)

    # Refit committee on all train+val
    X_trainval, dv_idx = prepare_features(river_trainval, feature_cols)
    y_trainval = river_trainval[TARGET].to_numpy()

    final_committee = RiverCommittee(dollar_volume_col=dv_idx)
    final_committee.fit(X_trainval, y_trainval)
    trainval_raw_scores = final_committee.predict(X_trainval)

    X_holdout, _ = prepare_features(river_holdout, feature_cols)
    holdout_raw_scores = final_committee.predict(X_holdout)

    if champion_name == "ranking":
        # Apply tuned quantile thresholds using trainval score distribution
        fold_q, sbet_q = champion["quantiles"]
        fold_thresh = float(np.percentile(trainval_raw_scores, fold_q * 100))
        sbet_thresh = float(np.percentile(trainval_raw_scores, sbet_q * 100))
        holdout_actions = []
        for s in holdout_raw_scores:
            if s >= fold_thresh:
                holdout_actions.append("FOLD")
            elif s >= sbet_thresh:
                holdout_actions.append("SMALL_BET")
            else:
                holdout_actions.append("SIZE_UP")
        holdout_bss = None
    else:
        # Refit calibrator on all OOF scores, predict on holdout
        final_calibrator = _FITTERS[champion_name](oof_scores, oof_adverse_labels)
        holdout_p_tail = _PREDICTORS[champion_name](final_calibrator, holdout_raw_scores)

        holdout_bs = brier_score(holdout_p_tail, holdout_adverse)
        holdout_base = holdout_adverse.mean()
        holdout_climo = brier_score(
            np.full_like(holdout_adverse, holdout_base, dtype=float), holdout_adverse,
        )
        holdout_bss = (1 - holdout_bs / holdout_climo) if holdout_climo > 0 else 0.0

        holdout_actions = assign_actions(holdout_p_tail, champion["thresholds"])
        print(f"  Holdout BSS:        {holdout_bss:.3f}")
        print(f"  Holdout p_tail:     [{holdout_p_tail.min():.3f}, {holdout_p_tail.max():.3f}]")
        print(f"  Holdout unique:     {len(np.unique(np.round(holdout_p_tail, 4)))}")

    holdout_report = coverage_report(holdout_actions, holdout_severe, holdout_adverse)

    print(f"  Holdout N:          {len(holdout_adverse)}")
    print(f"  Holdout FSR:        {holdout_report['false_safe_rate']:.1%}")
    print(f"  Holdout SIZE_UP:    {holdout_report['size_up_pct']:.1%}")
    print(f"  Holdout SMALL_BET:  {holdout_report['small_bet_pct']:.1%}")
    print(f"  Holdout FOLD:       {holdout_report['fold_pct']:.1%}")
    print(f"  Holdout detection:  {holdout_report['adverse_detection_rate']:.1%}")
    print(f"  Holdout mix:        {summarize_actions(holdout_actions)}")

    # ================================================================
    # MANIFEST
    # ================================================================
    manifest = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": SEED,
        "split": {"trainval_frac": 0.70, "holdout_frac": 0.30},
        "sizes": {
            "river_trainval": river_trainval.height,
            "river_holdout": river_holdout.height,
        },
        "tuning_target": {"max_fsr": MAX_FSR_TUNE, "min_sizeup": MIN_SIZE_UP_PCT},
        "challengers": {
            name: {
                k: (v if not isinstance(v, ActionThresholds) else
                    {"fold_min": v.fold_min, "small_bet_min": v.small_bet_min})
                for k, v in c.items()
            }
            for name, c in challengers.items()
        },
        "champion": champion_name,
        "holdout_bss": holdout_bss,
        "holdout_fsr": holdout_report["false_safe_rate"],
        "holdout_detection": holdout_report["adverse_detection_rate"],
        "holdout_mix": summarize_actions(holdout_actions),
    }

    manifest_path = Path("data/challenger_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
