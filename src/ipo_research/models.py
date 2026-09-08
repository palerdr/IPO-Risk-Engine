"""Fit a fixed linear classifier and a shallow boosted-tree challenger."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import FEATURES

MODEL_CONFIG = {
    "logistic": {"C": 0.1, "max_iter": 2000, "solver": "lbfgs", "random_state": 42},
    "boosted_trees": {
        "n_estimators": 75,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "subsample": 1.0,
        "n_iter_no_change": None,
        "random_state": 42,
    },
}


def matrix(rows: list[dict]) -> np.ndarray:
    result = np.array([[row["features"][key] for key in FEATURES] for row in rows], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("Model features must be finite")
    return result


def fit_models(rows: list[dict]) -> dict:
    x = matrix(rows)
    y = np.array([row["event"] for row in rows])
    if len(set(y)) != 2:
        raise ValueError("Training requires both event classes")
    logistic = make_pipeline(StandardScaler(), LogisticRegression(**MODEL_CONFIG["logistic"]))
    trees = GradientBoostingClassifier(**MODEL_CONFIG["boosted_trees"])
    return {"logistic": logistic.fit(x, y), "boosted_trees": trees.fit(x, y)}


def export_logistic(model, train: list[dict]) -> dict:
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    return {
        "features": list(FEATURES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "train_symbols": [row["symbol"] for row in train],
        "trained_through": max(row["label_end"] for row in train),
    }


def predict_frozen(model: dict, features: dict, as_of: str) -> tuple[float, list[dict]]:
    if model["trained_through"] >= as_of:
        raise ValueError("Training outcomes must finish before the prediction date")
    if model["features"] != list(FEATURES):
        raise ValueError("Model feature schema does not match")
    values = np.array([features[key] for key in FEATURES], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction features must be finite")
    contributions = (values - model["mean"]) / model["scale"] * model["coefficients"]
    score = float(model["intercept"] + contributions.sum())
    probability = float(1 / (1 + np.exp(-np.clip(score, -700, 700))))
    drivers = [
        {"feature": key, "value": float(value), "log_odds_contribution": float(contribution)}
        for key, value, contribution in zip(FEATURES, values, contributions)
    ]
    return probability, sorted(
        drivers, key=lambda row: abs(row["log_odds_contribution"]), reverse=True
    )
