"""Evaluate fixed models on future date blocks and save an auditable replay report."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .dataset import (
    EVENT_THRESHOLD,
    FEATURES,
    HORIZON_SESSIONS,
    OBSERVATION_SESSIONS,
    build_dataset,
)
from .models import MODEL_CONFIG, export_logistic, fit_models, matrix, predict_frozen

MODEL_NAMES = {
    "base_rate": "Training event rate",
    "logistic": "Logistic regression",
    "boosted_trees": "Shallow boosted trees",
}


def temporal_folds(rows: list[dict], count: int = 3) -> list[dict]:
    if len({row["symbol"] for row in rows}) != len(rows):
        raise ValueError("Keep one observation per issuer")
    dates = sorted({row["as_of"] for row in rows})
    if len(dates) < 12:
        raise ValueError("Need at least 12 distinct observation dates")
    evaluation_dates = dates[len(dates) // 2 :]
    result = []
    for index, block in enumerate(np.array_split(evaluation_dates, count)):
        if not len(block):
            continue
        start, end = str(block[0]), str(block[-1])
        candidates = [row for row in rows if row["as_of"] < start]
        train = [row for row in candidates if row["label_end"] < start]
        test = [row for row in rows if start <= row["as_of"] <= end]
        if len(train) < 20 or len({row["event"] for row in train}) < 2:
            raise ValueError(
                f"Fold {index + 1} needs 20 mature training observations and both classes"
            )
        result.append(
            {
                "id": index + 1,
                "start": start,
                "end": end,
                "train": train,
                "test": test,
                "purged": len(candidates) - len(train),
            }
        )
    return result


def screening(y: np.ndarray, p: np.ndarray, severe: np.ndarray, threshold: float) -> dict:
    flagged = p >= threshold
    events, flags, severe_count = int(y.sum()), int(flagged.sum()), int(severe.sum())
    true_positive = int((flagged & y.astype(bool)).sum())
    return {
        "threshold": threshold,
        "flagged": flags,
        "flagged_share": float(flagged.mean()),
        "events": events,
        "caught": true_positive,
        "precision": true_positive / flags if flags else None,
        "recall": true_positive / events if events else None,
        "severe_events": severe_count,
        "missed_severe": int((~flagged & severe).sum()),
        "missed_severe_rate": float((~flagged & severe).sum() / severe_count)
        if severe_count
        else None,
    }


def skill_interval(
    y: np.ndarray, p: np.ndarray, baseline: np.ndarray, months: list[str]
) -> list[float] | None:
    groups = sorted(set(months))
    if len(groups) < 3:
        return None
    indices = [np.flatnonzero(np.array(months) == month) for month in groups]
    rng = np.random.default_rng(42)
    skills = []
    for _ in range(500):
        selected = np.concatenate([indices[i] for i in rng.integers(0, len(groups), len(groups))])
        reference = np.mean((baseline[selected] - y[selected]) ** 2)
        if reference > 0:
            skills.append(1 - np.mean((p[selected] - y[selected]) ** 2) / reference)
    return np.quantile(skills, [0.025, 0.975]).tolist() if skills else None


def metrics(predictions: list[dict], model: str, intervals: bool = True) -> dict:
    y = np.array([row["event"] for row in predictions])
    p = np.array([row["probabilities"][model] for row in predictions])
    baseline = np.array([row["probabilities"]["base_rate"] for row in predictions])
    severe = np.array([row["drawdown"] >= 0.3 for row in predictions])
    reference = float(brier_score_loss(y, baseline))
    bs = float(brier_score_loss(y, p))
    bins = []
    for left, right in zip([0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 1]):
        mask = (p >= left) & ((p <= right) if right == 1 else (p < right))
        if mask.any():
            bins.append(
                {
                    "lower": left,
                    "upper": right,
                    "n": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "event_rate": float(y[mask].mean()),
                }
            )
    return {
        "n": len(y),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "brier": bs,
        "brier_skill": 1 - bs / reference if reference > 0 else None,
        "brier_skill_interval": skill_interval(
            y, p, baseline, [row["as_of"][:7] for row in predictions]
        )
        if intervals
        else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "calibration": bins,
        "screening": [screening(y, p, severe, cutoff) for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5]],
    }


def run_evaluation(input_path: Path, output_path: Path) -> dict:
    raw = input_path.read_bytes()
    bundle = json.loads(raw)
    rows, exclusions = build_dataset(bundle)
    predictions, reports, model_snapshots = [], [], []
    for fold in temporal_folds(rows):
        train, test = fold["train"], fold["test"]
        models = fit_models(train)
        frozen = export_logistic(models["logistic"], train)
        model_snapshots.append({"fold": fold["id"], **frozen})
        x = matrix(test)
        probabilities = {name: model.predict_proba(x)[:, 1] for name, model in models.items()}
        base_rate = float(np.mean([row["event"] for row in train]))
        fold_predictions = []
        for i, row in enumerate(test):
            predicted, drivers = predict_frozen(frozen, row["features"], row["as_of"])
            if not np.isclose(predicted, probabilities["logistic"][i], atol=1e-12):
                raise ValueError("Frozen model disagrees with fitted model")
            fold_predictions.append(
                {
                    **row,
                    "fold": fold["id"],
                    "drivers": drivers,
                    "probabilities": {
                        "base_rate": base_rate,
                        **{name: float(values[i]) for name, values in probabilities.items()},
                    },
                }
            )
        predictions.extend(fold_predictions)
        reports.append(
            {
                "id": fold["id"],
                "test_start": fold["start"],
                "test_end": fold["end"],
                "train_count": len(train),
                "test_count": len(test),
                "purged_count": fold["purged"],
                "train_label_end": frozen["trained_through"],
                "train_events": sum(row["event"] for row in train),
                "models": {
                    name: metrics(fold_predictions, name, intervals=False) for name in MODEL_NAMES
                },
            }
        )
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    report = {
        "schema_version": "2.0.0",
        "primary_model": "logistic",
        "synthetic": False,
        "target": {
            "observation_sessions": OBSERVATION_SESSIONS,
            "horizon_sessions": HORIZON_SESSIONS,
            "drawdown_threshold": EVENT_THRESHOLD,
            "definition": "Maximum adjusted-close peak-to-trough decline over sessions 21–40, including the session-20 close as the initial peak.",
        },
        "provenance": {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "universe_sha256": bundle["universe_sha256"],
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
            },
            "packages": {name: version(name) for name in ["numpy", "scikit-learn"]},
            "retrieved_at": max(
                [bundle["benchmark"]["retrieved_at"]]
                + [row["retrieved_at"] for row in bundle["listings"]]
            ),
            "benchmark_source": bundle["benchmark"]["source_page"],
        },
        "cohort": {
            "requested": len(bundle["listings"]) + len(bundle["exclusions"]),
            "eligible": len(rows),
            "evaluated": len(predictions),
            "exclusions": exclusions,
            "selection": bundle["selection"],
            "observation_start": rows[0]["as_of"],
            "observation_end": rows[-1]["as_of"],
        },
        "features": list(FEATURES),
        "model_config": MODEL_CONFIG,
        "protocol": {
            "selection": "Primary model and hyperparameters fixed before evaluation; no winner selection on test results.",
            "split": "Three expanding date blocks. Training label_end must precede the first test observation. One snapshot per issuer.",
            "calibration": "No post-hoc calibrator. Inspect probability calibration on held-out predictions; no claim of calibrated probabilities.",
            "uncertainty": "Exploratory 95% interval from 500 paired observation-month bootstrap samples of held-out predictions. Does not include model-fit or sample-selection uncertainty; overlapping horizons can cross months.",
            "screening": "Display fixed cutoffs from 10% to 50%; no optimized risk cap or position sizing.",
        },
        "limitations": bundle["limitations"]
        + [
            "Small cohort with overlapping market exposure. These results do not establish performance on the IPO population.",
            "The study evaluates risk forecasts. It does not simulate trade execution, costs, or portfolio returns.",
            "The price-only MVP excludes filing features until point-in-time availability can be audited.",
        ],
        "models": [
            {"id": name, "name": title, **metrics(predictions, name)}
            for name, title in MODEL_NAMES.items()
        ],
        "folds": reports,
        "predictions": predictions,
        "frozen_logistic_models": model_snapshots,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
