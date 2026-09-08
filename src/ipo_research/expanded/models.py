"""Train fixed baselines and select challengers within mature training periods."""

import io
import json
import pickle
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..dataset import FEATURES as PRICE_FEATURES
from ..models import MODEL_CONFIG
from .dataset import PRE_CATEGORICAL, PRE_FEATURES
from .storage import save_json, sha


class BrierMetric(Metric):
    def __init__(self):
        self._name = "brier"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return float(np.mean((y_score[:, 1] - y_true) ** 2))


def labels(rows):
    return np.array([r["event"] for r in rows], dtype=int)


def features_for(stage: int, view="full") -> tuple[str, ...]:
    if view == "price":
        if stage != 20:
            raise ValueError("Price features require a day-20 observation")
        return tuple(PRICE_FEATURES)
    return PRE_FEATURES + (tuple(PRICE_FEATURES) if stage else ())


def cat_matrix(rows, features):
    return [
        [
            row["features"][key] if row["features"][key] is not None else float("nan")
            for key in features
        ]
        for row in rows
    ]


@dataclass
class Transform:
    numeric: list
    categorical: list
    median: list
    mean: list
    scale: list
    categories: dict

    @classmethod
    def fit(cls, rows, features):
        numeric = [key for key in features if key not in PRE_CATEGORICAL]
        categorical = [key for key in features if key in PRE_CATEGORICAL]
        x = np.array([[row["features"][key] for key in numeric] for row in rows], dtype=float)
        medians = [
            float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
            for column in x.T
        ]
        x = np.where(np.isfinite(x), x, medians)
        scale = x.std(axis=0)
        scale[scale < 1e-12] = 1
        categories = {key: sorted({row["features"][key] for row in rows}) for key in categorical}
        return cls(
            numeric, categorical, medians, x.mean(axis=0).tolist(), scale.tolist(), categories
        )

    def apply(self, rows, embedding=False):
        x = np.array([[row["features"][key] for key in self.numeric] for row in rows], dtype=float)
        missing = ~np.isfinite(x)
        x = np.where(missing, self.median, x)
        pieces = [(x - self.mean) / self.scale, missing.astype(float)]
        for key in self.categorical:
            mapping = {value: i + 1 for i, value in enumerate(self.categories[key])}
            codes = np.array([mapping.get(row["features"][key], 0) for row in rows])
            pieces.append(codes[:, None] if embedding else np.eye(len(mapping) + 1)[codes])
        result = np.column_stack(pieces).astype(np.float32 if embedding else np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite transformed features")
        return result


def inner_split(rows, fraction=0.2):
    dates = sorted({row["as_of"] for row in rows})
    cutoff = dates[int(len(dates) * (1 - fraction))]
    fit = [row for row in rows if row["label_end"] < cutoff]
    validation = [row for row in rows if row["as_of"] >= cutoff]
    if (
        len(fit) < 50
        or len(validation) < 20
        or len(set(labels(fit))) != 2
        or len(set(labels(validation))) != 2
    ):
        raise ValueError(
            "Inner split needs mature training and validation examples of both classes"
        )
    if {r["issuer_id"] for r in fit} & {r["issuer_id"] for r in validation}:
        raise ValueError("Issuer leakage in the inner split")
    return (
        fit,
        validation,
        {
            "start": cutoff,
            "fit_count": len(fit),
            "validation_count": len(validation),
            "fit_label_end": max(r["label_end"] for r in fit),
            "purged": len(rows) - len(fit) - len(validation),
            "fit_ids": [r["id"] for r in fit],
            "validation_ids": [r["id"] for r in validation],
        },
    )


def fit_baseline(train, test, stage, kind, full=False):
    features = features_for(stage, "full" if stage == 0 or full else "price")
    if stage and not full:
        x = np.array(cat_matrix(train, features), dtype=float)
        xt = np.array(cat_matrix(test, features), dtype=float)
        transform = None
        model = (
            make_pipeline(StandardScaler(), LogisticRegression(**MODEL_CONFIG["logistic"]))
            if kind == "logistic"
            else GradientBoostingClassifier(**MODEL_CONFIG["boosted_trees"])
        )
    else:
        transform = Transform.fit(train, features)
        x, xt = transform.apply(train), transform.apply(test)
        model = (
            LogisticRegression(**MODEL_CONFIG["logistic"])
            if kind == "logistic"
            else GradientBoostingClassifier(**MODEL_CONFIG["boosted_trees"])
        )
    model.fit(x, labels(train))
    return model.predict_proba(xt)[:, 1], {
        "model": model,
        "transform": transform,
        "features": features,
    }


def cat_model(config, protocol, seed, iterations):
    return CatBoostClassifier(
        **config,
        iterations=iterations,
        learning_rate=protocol["catboost_learning_rate"],
        loss_function="Logloss",
        eval_metric="BrierScore",
        boosting_type="Ordered",
        random_seed=seed,
        thread_count=2,
        allow_writing_files=False,
        verbose=False,
    )


def train_catboost(train, test, features, protocol, target: Path):
    fit, validation, split = inner_split(train, protocol["inner_validation_fraction_dates"])
    categorical = [i for i, key in enumerate(features) if key in PRE_CATEGORICAL]
    trials = []
    for config in protocol["catboost_candidates"]:
        model = cat_model(config, protocol, protocol["seeds"][0], protocol["catboost_iterations"])
        model.fit(
            cat_matrix(fit, features),
            labels(fit),
            cat_features=categorical,
            eval_set=(cat_matrix(validation, features), labels(validation)),
            early_stopping_rounds=protocol["catboost_early_stopping_rounds"],
        )
        prediction = model.predict_proba(cat_matrix(validation, features))[:, 1]
        trials.append(
            {
                "config": config,
                "iterations": int(model.tree_count_),
                "validation_brier": float(brier_score_loss(labels(validation), prediction)),
            }
        )
    winner = min(trials, key=lambda trial: trial["validation_brier"])
    predictions, files = [], []
    for seed in protocol["seeds"]:
        model = cat_model(winner["config"], protocol, seed, winner["iterations"])
        model.fit(cat_matrix(train, features), labels(train), cat_features=categorical)
        predictions.append(model.predict_proba(cat_matrix(test, features))[:, 1])
        file = target / f"seed-{seed}.cbm"
        model.save_model(str(file))
        files.append(file.name)
    return np.mean(predictions, axis=0), {
        "inner_split": split,
        "trials": trials,
        "selected": winner,
        "seed_predictions": [p.tolist() for p in predictions],
        "files": files,
    }


def tab_model(transform, config, protocol, seed):
    cat_start = 2 * len(transform.numeric)
    return TabNetClassifier(
        **config,
        cat_idxs=list(range(cat_start, cat_start + len(transform.categorical))),
        cat_dims=[len(transform.categories[key]) + 1 for key in transform.categorical],
        cat_emb_dim=2,
        seed=seed,
        verbose=0,
        device_name="cpu",
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": protocol["tabnet_learning_rate"], "weight_decay": 1e-5},
    )


def tab_fit(model, transform, train, protocol, epochs, validation=None):
    batch_size = min(protocol["tabnet_batch_size"], len(train))
    while len(train) % batch_size == 1:
        batch_size -= 1
    kwargs = {
        "max_epochs": epochs,
        "batch_size": batch_size,
        "virtual_batch_size": protocol["tabnet_virtual_batch_size"],
        "num_workers": 0,
        "drop_last": False,
        "weights": 0,
        "compute_importance": False,
        "patience": protocol["tabnet_patience"] if validation else 0,
    }
    if validation:
        kwargs.update(
            eval_set=[(transform.apply(validation, embedding=True), labels(validation))],
            eval_name=["validation"],
            eval_metric=[BrierMetric],
        )
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Early stopping.*")
        warnings.filterwarnings("ignore", message=".*best weights.*")
        model.fit(transform.apply(train, embedding=True), labels(train), **kwargs)


def train_tabnet(train, test, features, protocol, target: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit, validation, split = inner_split(train, protocol["inner_validation_fraction_dates"])
    transform = Transform.fit(fit, features)
    trials = []
    for config in protocol["tabnet_candidates"]:
        model = tab_model(transform, config, protocol, protocol["seeds"][0])
        tab_fit(model, transform, fit, protocol, protocol["tabnet_max_epochs"], validation)
        p = model.predict_proba(transform.apply(validation, embedding=True))[:, 1]
        trials.append(
            {
                "config": config,
                "epochs": int(model.best_epoch) + 1,
                "epochs_run": len(model.history["loss"]),
                "validation_brier": float(brier_score_loss(labels(validation), p)),
            }
        )
    winner = min(trials, key=lambda trial: trial["validation_brier"])
    transform = Transform.fit(train, features)
    save_json(target / "transform.json", transform.__dict__)
    predictions, files = [], ["transform.json"]
    for seed in protocol["seeds"]:
        model = tab_model(transform, winner["config"], protocol, seed)
        tab_fit(model, transform, train, protocol, winner["epochs"])
        predictions.append(
            model.predict_proba(transform.apply(test, embedding=True))[:, 1].astype(float)
        )
        file = target / f"seed-{seed}"
        with redirect_stdout(io.StringIO()):
            model.save_model(str(file))
        files.append(file.name + ".zip")
    return np.mean(predictions, axis=0), {
        "inner_split": split,
        "trials": trials,
        "selected": winner,
        "seed_predictions": [p.tolist() for p in predictions],
        "files": files,
        "inference_batch_size": model.batch_size,
    }


def train_model(name, train, test, stage, protocol, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    full = name == "logistic_enriched"
    features = features_for(stage, "price" if name == "catboost_price" else "full")
    if name in ("logistic", "boosted_trees", "logistic_enriched"):
        prediction, fitted = fit_baseline(
            train, test, stage, "logistic" if full else name, full=full
        )
        file = target / "model.pkl"
        file.write_bytes(pickle.dumps(fitted))
        info = {"files": [file.name], "features": list(fitted["features"])}
    elif name in ("catboost", "catboost_price"):
        prediction, info = train_catboost(train, test, features, protocol, target)
        info["features"] = list(features)
    elif name == "tabnet":
        prediction, info = train_tabnet(train, test, features, protocol, target)
        info["features"] = list(features)
    else:
        raise ValueError(name)
    info.update(
        model=name,
        stage=stage,
        trained_through=max(r["label_end"] for r in train),
        train_ids=[r["id"] for r in train],
        train_issuers=[r["issuer_id"] for r in train],
        artifact_hashes={name: sha(target / name) for name in info["files"]},
    )
    save_json(target / "manifest.json", info)
    return prediction, info


def predict_saved(target: Path, rows):
    """Reload local, hash-checked artifacts and enforce the training cutoff."""
    info = json.loads((target / "manifest.json").read_text())
    if any(
        row["as_of"] <= info["trained_through"] or row["issuer_id"] in info["train_issuers"]
        for row in rows
    ):
        raise ValueError("Prediction overlaps the model's training period or issuers")
    for name, expected in info["artifact_hashes"].items():
        if sha(target / name) != expected:
            raise ValueError(f"Model artifact hash mismatch: {name}")
    if info["model"] in ("logistic", "boosted_trees", "logistic_enriched"):
        fitted = pickle.loads((target / "model.pkl").read_bytes())
        x = (
            fitted["transform"].apply(rows)
            if fitted["transform"]
            else np.array(cat_matrix(rows, fitted["features"]), dtype=float)
        )
        return fitted["model"].predict_proba(x)[:, 1]
    predictions = []
    if info["model"] == "tabnet":
        transform = Transform(**json.loads((target / "transform.json").read_text()))
        x = transform.apply(rows, embedding=True)
        for name in info["files"]:
            if name.endswith(".zip"):
                model = TabNetClassifier(device_name="cpu", verbose=0)
                model.load_model(str(target / name))
                model.batch_size = info["inference_batch_size"]
                predictions.append(model.predict_proba(x)[:, 1].astype(float))
    else:
        for name in info["files"]:
            model = CatBoostClassifier()
            model.load_model(str(target / name))
            predictions.append(model.predict_proba(cat_matrix(rows, info["features"]))[:, 1])
    return np.mean(predictions, axis=0)
