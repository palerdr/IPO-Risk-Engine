"""Compare model forecasts on shared future blocks and retain per-issuer evidence."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ..evaluate import screening
from .models import train_model, predict_saved
from .sources import load_json, save_json, sha


def temporal_folds(rows, protocol):
    if len({r["issuer_id"] for r in rows}) != len(rows):
        raise ValueError("Use one observation per issuer within each stage")
    folds = []
    for index, (first, last) in enumerate(protocol["outer_test_years"], 1):
        start, end = f"{first}-01-01", f"{last}-12-31"
        train = [row for row in rows if row["label_end"] < start
                 and row["listing_date"] >= protocol.get("training_listing_start", "0000-01-01")]
        test = [row for row in rows if start <= row["as_of"] <= end]
        if len(train) < 100 or not test or len({r["event"] for r in train}) < 2:
            raise ValueError(f"Insufficient data for fold {index}")
        if {r["issuer_id"] for r in train} & {r["issuer_id"] for r in test}:
            raise ValueError("Outer train/test issuer overlap")
        folds.append({"id": index, "start": start, "end": end, "train": train, "test": test,
                      "purged": sum(r["as_of"] < start <= r["label_end"] for r in rows)})
    return folds


def stats(rows, name):
    y = np.array([r["event"] for r in rows])
    p = np.array([r["probabilities"][name] for r in rows])
    base = np.array([r["probabilities"]["base_rate"] for r in rows])
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid forecast probabilities")
    brier, reference = float(brier_score_loss(y, p)), float(brier_score_loss(y, base))
    bins = []
    for i in range(5):
        left, right = i / 5, (i + 1) / 5
        mask = (p >= left) & ((p < right) if i < 4 else (p <= right))
        if mask.any():
            bins.append({"lower": left, "upper": right, "n": int(mask.sum()),
                         "probability": float(p[mask].mean()), "event_rate": float(y[mask].mean())})
    return {"n": len(rows), "events": int(y.sum()), "event_rate": float(y.mean()),
            "brier": brier, "brier_skill": 1 - brier / reference,
            "brier_skill_vs_constant_50": 1 - brier / .25,
            "training_rate_reference_brier": reference, "constant_50_reference_brier": .25,
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_probability": float(p.mean()), "calibration": bins,
            "screening": [screening(y, p, np.array([r["drawdown"] >= .3 for r in rows]), cutoff)
                          for cutoff in (.2, .3, .5)]}


def paired_intervals(rows, name, replicates=2000):
    y = np.array([r["event"] for r in rows])
    p = np.array([r["probabilities"][name] for r in rows])
    base = np.array([r["probabilities"]["base_rate"] for r in rows])
    logistic = np.array([r["probabilities"]["logistic"] for r in rows])
    quarters = np.array([f"{r['as_of'][:4]}-{(int(r['as_of'][5:7]) - 1) // 3}" for r in rows])
    indices = [np.flatnonzero(quarters == quarter) for quarter in sorted(set(quarters))]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(replicates):
        selected = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
        loss = np.mean((p[selected] - y[selected]) ** 2)
        reference = np.mean((base[selected] - y[selected]) ** 2)
        linear = np.mean((logistic[selected] - y[selected]) ** 2)
        samples.append([loss, 1 - loss / reference, loss - linear, 1 - loss / .25])
    limits = np.quantile(samples, [.025, .975], axis=0)
    return {key: limits[:, i].tolist() for i, key in enumerate(("brier_ci", "brier_skill_ci", "brier_difference_vs_logistic_ci", "brier_skill_vs_constant_50_ci"))}


def train(directory: Path, artifact_dir: Path, stages=(0, 20), *, sensitivity=False) -> dict:
    data = load_json(directory / "dataset.json.gz")
    protocol_path = directory / ("coverage-protocol.json" if sensitivity else "protocol.json")
    protocol = load_json(protocol_path)
    if sensitivity:
        artifact_dir = artifact_dir / "coverage-2018"
    root = Path(__file__).resolve().parents[3]
    old_input = root / "research/input.json"
    old_rows = json.loads(old_input.read_text())["listings"] if old_input.exists() else []
    old_cases = {(r["symbol"], r["listing_date"]) for r in old_rows}
    source = Path(__file__).parent
    training_hashes = {str(p.relative_to(source.parent)): sha(p) for p in
                       [source / "models.py", source / "dataset.py", source.parent / "models.py", source.parent / "dataset.py"]}
    provenance = {"dataset_sha256": sha(directory / "dataset.json.gz"), "protocol_sha256": sha(protocol_path),
                  "training_source_sha256": training_hashes,
                  "packages": {name: version(name) for name in ("numpy", "scikit-learn", "catboost", "pytorch-tabnet", "torch")}}
    reports = {}
    for stage in stages:
        rows = data["stages"][str(stage)]
        names = ["logistic", "boosted_trees", "catboost", "tabnet"]
        if stage:
            names = ["logistic", "boosted_trees", "logistic_enriched", "catboost_price", "catboost", "tabnet"]
        predictions, folds = [], []
        for fold in temporal_folds(rows, protocol):
            train_rows, test = fold["train"], fold["test"]
            probability = {"base_rate": np.full(len(test), np.mean([r["event"] for r in train_rows]))}
            model_reports = {}
            for name in names:
                target = artifact_dir / f"stage-{stage}" / f"fold-{fold['id']}" / name
                checkpoint = target / "result.json"
                cache_key = {**provenance, "train_ids": [r["id"] for r in train_rows], "test_ids": [r["id"] for r in test]}
                started = time.monotonic()
                cached = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
                if cached and cached["cache_key"] == cache_key:
                    p = predict_saved(target, test)
                    np.testing.assert_allclose(p, cached["probabilities"], atol=1e-7)
                    info = cached["info"]
                else:
                    p, info = train_model(name, train_rows, test, stage, protocol, target)
                    reloaded = predict_saved(target, test)
                    np.testing.assert_allclose(p, reloaded, atol=1e-7)
                    save_json(checkpoint, {"cache_key": cache_key, "probabilities": p.tolist(), "info": info})
                probability[name] = p
                model_reports[name] = {key: value for key, value in info.items()
                                       if key not in ("seed_predictions", "train_ids", "train_issuers")}
                model_reports[name]["artifact_path"] = str(target)
                if info.get("seed_predictions"):
                    model_reports[name]["seed_brier"] = [float(brier_score_loss([r["event"] for r in test], values))
                                                          for values in info["seed_predictions"]]
                print(f"Stage {stage}, fold {fold['id']}, {name}: fit/reload verified in {time.monotonic()-started:.1f}s", flush=True)
            records = [{**{key: value for key, value in row.items() if key != "features"},
                        "fold": fold["id"], "prior_mvp_case": (row["symbol"], row["listing_date"]) in old_cases,
                        "probabilities": {name: float(values[i]) for name, values in probability.items()}}
                       for i, row in enumerate(test)]
            predictions.extend(records)
            folds.append({"id": fold["id"], "test_start": fold["start"], "test_end": fold["end"],
                          "train_count": len(train_rows), "test_count": len(test), "purged_count": fold["purged"],
                          "train_label_end": max(r["label_end"] for r in train_rows),
                          "train_event_rate": float(probability["base_rate"][0]),
                          "models": {name: stats(records, name) for name in probability}, "training": model_reports})
            save_json(directory / f"stage-{stage}-progress.json", {"folds": folds, "predictions": predictions})
        metrics = {name: {**stats(predictions, name), **paired_intervals(predictions, name, protocol["bootstrap_replicates"])}
                   for name in ["base_rate"] + names}
        fresh = [r for r in predictions if not r["prior_mvp_case"]]
        recent = [r for r in predictions if r["as_of"] >= "2022-01-01"]
        reports[str(stage)] = {"eligible": len(rows), "models": metrics, "folds": folds, "predictions": predictions,
                               "without_prior_mvp": {name: stats(fresh, name) for name in metrics},
                               "recent_2022_2025": {name: stats(recent, name) for name in metrics}}
    provenance["evaluation_source_sha256"] = sha(Path(__file__))
    result = {"schema_version": "3.0.0", "generated_at": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance, "protocol": protocol, "quality": data["quality"], "stages": reports,
              "limitations": [
                  "The registry includes missing and delisted tickers, but Yahoo history coverage limits the measured cohort. Missing outcomes can cause survivorship bias.",
                  "Historical issuer traits and offer terms come from current research compilations. Sources describe IPO-time facts; they do not provide archived publication timestamps for each field.",
                  "Name and unit-ticker rules screen acquisition companies. These rules do not establish a complete audited security-type classification.",
                  "The pre-IPO model uses offer terms and issuer traits. It excludes filing financial statements until dated extraction is available.",
                  "Bootstrap intervals condition on fitted predictions and the retained cohort. They exclude model-selection and missing-outcome uncertainty.",
                  "The experiments estimate drawdown risk. They do not establish trading returns, execution quality, or offer-price investment returns."]}
    save_json(directory / ("coverage-results.json.gz" if sensitivity else "results.json.gz"), result)
    return result
