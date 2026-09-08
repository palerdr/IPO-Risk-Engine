"""Run the risk-model stages on synthetic data without credentials or downloads."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import polars as pl

from ipo_risk_engine.dataset.assemble import snapshots_to_dataframe, temporal_train_val_test_split
from ipo_risk_engine.features.street_features import compute_daily_street_features
from ipo_risk_engine.labels.mdd import compute_forward_mdd
from ipo_risk_engine.models.calibrate import build_oof_calibration_frame, select_calibrator_oof, predict_p_tail
from ipo_risk_engine.models.committee import RiverCommittee, prepare_features
from ipo_risk_engine.policy.actions import assign_actions, summarize_actions
from ipo_risk_engine.snapshots.builder import Snapshot


def run_demo() -> dict:
    rng = np.random.default_rng(42)
    snapshots = []
    for index in range(80):
        volatility = .005 + (index % 8) * .01
        closes = 20 * np.exp(np.cumsum(rng.normal(0, volatility, 81)))
        opens = np.r_[20., closes[:-1]]
        start = datetime(2000, 1, 3, tzinfo=timezone.utc) + timedelta(days=index * 90)
        days = [start + timedelta(days=day) for day in range(120)]
        sessions = [day for day in days if day.weekday() < 5][:81]
        bars = pl.DataFrame({
            "ts": sessions, "open": opens, "close": closes,
            "high": np.maximum(opens, closes) * 1.01,
            "low": np.minimum(opens, closes) * .99,
            "volume": np.full(81, 100000, dtype=np.int64),
        })
        features = {}
        for street, offset, length in [("FLOP", 0, 6), ("TURN", 6, 15), ("RIVER", 21, 40)]:
            features.update(compute_daily_street_features(bars.slice(offset, length), street))
        label = compute_forward_mdd(bars, 20)["forward_mdd_20d"][60]
        snapshots.append(Snapshot(
            symbol=f"SYNTHETIC_{index:02d}", street="RIVER", asof_date=sessions[60].date(),
            ipo_date=sessions[0].date(), sector="synthetic", features=features,
            labels={"forward_mdd_20d": label},
        ))

    data = snapshots_to_dataframe(snapshots)
    train, validation, test = temporal_train_val_test_split(data)
    feature_names = list(snapshots[0].features)
    oof = build_oof_calibration_frame(train, feature_names, label_col="adverse_20")
    calibration = select_calibrator_oof(oof["score_oof"], oof["labels"])
    training_x, volume_index = prepare_features(train, feature_names)
    model = RiverCommittee(dollar_volume_col=volume_index)
    model.fit(training_x, train["risk_severity"].to_numpy())
    test_x, _ = prepare_features(test, feature_names)
    probabilities = predict_p_tail(calibration, model.predict(test_x))
    actions = assign_actions(probabilities)
    return {
        "data": "Synthetic prices; this run demonstrates code execution, not investment performance.",
        "seed": 42,
        "rows": {"train": train.height, "validation": validation.height, "test": test.height},
        "features": len(feature_names),
        "calibrator": calibration.selected,
        "policy": "Fixed default thresholds; no claim of a validated risk constraint.",
        "actions": summarize_actions(actions),
        "example": {
            "symbol": test["symbol"][0],
            "adverse_probability": round(float(probabilities[0]), 6),
            "action": actions[0],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
