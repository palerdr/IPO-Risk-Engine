"""
Calibrator selection: map committee risk_severity predictions → p(adverse event).

Three calibrators compared via OOF temporal expanding folds on train+val:
  1. Platt (sigmoid / logistic regression on 1D scores)
  2. Isotonic regression
  3. Binned empirical with Laplace smoothing (4-5 monotone bins)

Selection by out-of-fold Brier Skill Score (BSS) vs climatology.
Final calibrator is refit on full train+val, tested once on held-out test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ipo_risk_engine.models.committee import RiverCommittee, prepare_features


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    return float(np.mean((probs - labels) ** 2))


@dataclass
class CalibrationResult:
    """Holds the selected calibrator and comparison metrics."""
    selected: str                    # "platt", "isotonic", or "binned"
    calibrator: object               # fitted model (or BinnedCalibrator)
    brier_by_method: dict[str, float] = field(default_factory=dict)
    bss_by_method: dict[str, float] = field(default_factory=dict)
    brier_climatology: float = 0.0
    base_rate: float = 0.0


def _fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    X = scores.reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X, labels)
    return model


def _predict_platt(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores.reshape(-1, 1))[:, 1]


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    model.fit(scores, labels)
    return model


def _predict_isotonic(model: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict(scores)


class BinnedCalibrator:
    """Monotone binned calibrator with Laplace smoothing.

    Splits scores into n_bins quantile bins, computes empirical
    P(event) per bin with Laplace smoothing (add alpha pseudo-counts),
    then enforces monotonicity via pool-adjacent-violators.
    """

    def __init__(self, n_bins: int = 4, alpha: float = 1.0):
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_edges_: np.ndarray | None = None
        self.bin_probs_: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> BinnedCalibrator:
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(scores, percentiles)
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bin_edges_ = edges

        bin_probs = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            mask = (scores >= edges[i]) & (scores < edges[i + 1])
            if i == self.n_bins - 1:
                mask = (scores >= edges[i]) & (scores <= edges[i + 1])
            n = mask.sum()
            pos = labels[mask].sum() if n > 0 else 0
            bin_probs[i] = (pos + self.alpha) / (n + 2 * self.alpha)

        self.bin_probs_ = self._isotonic_pass(bin_probs)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        assert self.bin_edges_ is not None and self.bin_probs_ is not None
        indices = np.digitize(scores, self.bin_edges_[1:-1])
        indices = np.clip(indices, 0, self.n_bins - 1)
        return self.bin_probs_[indices]

    @staticmethod
    def _isotonic_pass(values: np.ndarray) -> np.ndarray:
        """Simple pool-adjacent-violators for ascending monotonicity."""
        result = values.copy().astype(float)
        n = len(result)
        i = 0
        while i < n - 1:
            if result[i] > result[i + 1]:
                j = i + 1
                while j < n and result[j] < result[i]:
                    j += 1
                avg = result[i:j].mean()
                result[i:j] = avg
            i += 1
        return result


def _fit_binned(scores: np.ndarray, labels: np.ndarray, n_bins: int = 4) -> BinnedCalibrator:
    return BinnedCalibrator(n_bins=n_bins).fit(scores, labels)


def _predict_binned(model: BinnedCalibrator, scores: np.ndarray) -> np.ndarray:
    return model.predict(scores)


_FITTERS = {"platt": _fit_platt, "isotonic": _fit_isotonic, "binned": _fit_binned}
_PREDICTORS = {"platt": _predict_platt, "isotonic": _predict_isotonic, "binned": _predict_binned}


def select_calibrator_oof(
        scores: np.ndarray,
        labels: np.ndarray,
        n_folds: int = 3,
) -> CalibrationResult:
    """Select calibrator by out-of-fold Brier Skill Score (BSS) vs climatology.

    Args:
        scores: Model scores to calibrate (e.g. predicted risk_severity).
        labels: Binary ground-truth labels (e.g. adverse event occurred).
        n_folds: Number of temporal expanding folds for OOF evaluation."""
    block_indices = np.array_split(np.arange(len(scores)), n_folds + 1)
    score_blocks = [scores[idx] for idx in block_indices]
    label_blocks = [labels[idx] for idx in block_indices]

    fold_preds = {name: np.zeros_like(labels, dtype=float) for name in _FITTERS}
    for k in range(1, n_folds + 1):
        train = np.concatenate(score_blocks[0:k])
        train_labels = np.concatenate(label_blocks[0:k])
        for name in _FITTERS:
            model = _FITTERS[name](train, train_labels)
            preds = _PREDICTORS[name](model, score_blocks[k])
            fold_preds[name][block_indices[k]] = preds

    oof_idx = np.concatenate(block_indices[1:])
    oof_labels = labels[oof_idx]
    brier_climo = brier_score(
        np.full_like(oof_labels, oof_labels.mean(), dtype=float), oof_labels
    )

    brier_by_method = {}
    bss_by_method = {}
    for name in _FITTERS:
        oof_preds = fold_preds[name][oof_idx]
        brier_by_method[name] = brier_score(oof_preds, oof_labels)
        bss_by_method[name] = (
            (1 - brier_by_method[name] / brier_climo) if brier_climo > 0 else 0.0
        )

    winner = max(bss_by_method, key=lambda k: bss_by_method[k])
    final_model = _FITTERS[winner](scores, labels)
    return CalibrationResult(
        selected=winner,
        calibrator=final_model,
        brier_by_method=brier_by_method,
        bss_by_method=bss_by_method,
        brier_climatology=brier_climo,
        base_rate=float(labels.mean()),
    )


def bootstrap_bss(
    scores: np.ndarray,
    labels: np.ndarray,
    calibrator_name: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for BSS of a given calibrator.

    Returns (bss_point, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    bss_samples = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, lab = scores[idx], labels[idx]
        if lab.sum() == 0 or lab.sum() == n:
            continue
        model = _FITTERS[calibrator_name](s, lab)
        preds = _PREDICTORS[calibrator_name](model, s)
        bs = brier_score(preds, lab)
        base = float(lab.mean())
        bs_climo = brier_score(np.full_like(lab, base, dtype=float), lab)
        if bs_climo > 0:
            bss_samples.append(1 - bs / bs_climo)

    bss_arr = np.array(bss_samples)
    return float(np.median(bss_arr)), float(np.percentile(bss_arr, 2.5)), float(np.percentile(bss_arr, 97.5))


def select_calibrator_hardened(
    oof_frame: dict[str, np.ndarray],
    n_folds: int = 3,
    bss_gap_threshold: float = 0.03,
    n_boot: int = 1000,
    seed: int = 42,
) -> CalibrationResult:
    """Hardened calibrator selection: OOF scores → OOF calibration → bootstrap CI.

    Selection rule:
      - Pick isotonic if BSS(isotonic) - BSS(binned) >= bss_gap_threshold
        AND bootstrap 95% CI for isotonic BSS excludes 0.
      - Otherwise fall back to binned (more stable with small N).
    """
    scores = oof_frame["score_oof"]
    labels = oof_frame["labels"]

    result = select_calibrator_oof(scores, labels, n_folds=n_folds)

    _, iso_lo, iso_hi = bootstrap_bss(scores, labels, "isotonic", n_boot, seed)
    _, bin_lo, bin_hi = bootstrap_bss(scores, labels, "binned", n_boot, seed)

    gap = result.bss_by_method.get("isotonic", 0.0) - result.bss_by_method.get("binned", 0.0)
    iso_ci_excludes_zero = iso_lo > 0

    if gap >= bss_gap_threshold and iso_ci_excludes_zero:
        winner = "isotonic"
    else:
        winner = "binned"

    final_model = _FITTERS[winner](scores, labels)

    return CalibrationResult(
        selected=winner,
        calibrator=final_model,
        brier_by_method=result.brier_by_method,
        bss_by_method=result.bss_by_method,
        brier_climatology=result.brier_climatology,
        base_rate=result.base_rate,
    )



def build_oof_calibration_frame(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "risk_severity",
    label_col: str = "adverse_event_15",
    n_splits: int = 3,
) -> dict[str, np.ndarray]:
    """Generate true OOF committee scores via expanding temporal folds.

    The dataframe must be temporally sorted (by asof_date).
    Each fold trains a fresh RiverCommittee on blocks 0..k-1,
    predicts on block k, and concatenates all OOF outputs.

    Returns dict with keys:
        score_oof  - committee predictions (only on OOF rows)
        y          - regression target for OOF rows
        labels     - binary adverse_event labels for OOF rows
        fold_id    - which fold each OOF row came from (1..n_splits)
    """
    df = df.sort("asof_date")
    block_indices = np.array_split(np.arange(df.height), n_splits + 1)

    X_full, dv_idx = prepare_features(df, feature_cols)
    y_full = df[target_col].to_numpy()
    labels_full = df[label_col].to_numpy().astype(float)

    oof_scores, oof_y, oof_labels, oof_folds = [], [], [], []
    for k in range(1, n_splits + 1):
        train_idx = np.concatenate(block_indices[0:k])
        val_idx = block_indices[k]

        comm = RiverCommittee(dollar_volume_col=dv_idx)
        comm.fit(X_full[train_idx], y_full[train_idx])
        preds = comm.predict(X_full[val_idx])

        oof_scores.append(preds)
        oof_y.append(y_full[val_idx])
        oof_labels.append(labels_full[val_idx])
        oof_folds.append(np.full(len(val_idx), k))

    return {
        "score_oof": np.concatenate(oof_scores),
        "y": np.concatenate(oof_y),
        "labels": np.concatenate(oof_labels),
        "fold_id": np.concatenate(oof_folds),
    }


def calibrate_oof_all_methods(
    scores: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 3,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Compute OOF calibrated predictions for ALL calibrator methods.

    Same expanding-fold logic as select_calibrator_oof, but returns
    every method's predictions instead of picking a winner.

    Returns:
        oof_preds_by_method: dict of method_name → OOF calibrated probabilities
        oof_idx: indices into the input arrays for OOF rows
        oof_labels: binary labels for OOF rows
        brier_climo: climatological Brier score for OOF rows
    """
    block_indices = np.array_split(np.arange(len(scores)), n_folds + 1)
    score_blocks = [scores[idx] for idx in block_indices]
    label_blocks = [labels[idx] for idx in block_indices]

    fold_preds = {name: np.zeros(len(scores), dtype=float) for name in _FITTERS}
    for k in range(1, n_folds + 1):
        train_scores = np.concatenate(score_blocks[0:k])
        train_labels = np.concatenate(label_blocks[0:k])
        for name in _FITTERS:
            model = _FITTERS[name](train_scores, train_labels)
            preds = _PREDICTORS[name](model, score_blocks[k])
            fold_preds[name][block_indices[k]] = preds

    oof_idx = np.concatenate(block_indices[1:])
    oof_labels = labels[oof_idx]
    brier_climo = brier_score(
        np.full_like(oof_labels, oof_labels.mean(), dtype=float), oof_labels
    )

    oof_preds_by_method = {
        name: fold_preds[name][oof_idx] for name in _FITTERS
    }

    return oof_preds_by_method, oof_idx, oof_labels, brier_climo


def select_calibrator_single(
    train_scores: np.ndarray,
    train_labels: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
) -> CalibrationResult:
    """Original single-holdout selection (kept for comparison)."""
    base_rate = float(train_labels.mean())
    climo_probs = np.full_like(val_labels, base_rate, dtype=float)
    brier_climo = brier_score(climo_probs, val_labels)

    brier_by_method = {}
    bss_by_method = {}
    fitted_models = {}

    for name in _FITTERS:
        model = _FITTERS[name](train_scores, train_labels)
        preds = _PREDICTORS[name](model, val_scores)
        bs = brier_score(preds, val_labels)
        bss = (1 - bs / brier_climo) if brier_climo > 0 else 0.0
        brier_by_method[name] = bs
        bss_by_method[name] = bss
        fitted_models[name] = model

    best = max(bss_by_method, key=lambda k: bss_by_method[k])

    return CalibrationResult(
        selected=best,
        calibrator=fitted_models[best],
        brier_by_method=brier_by_method,
        bss_by_method=bss_by_method,
        brier_climatology=brier_climo,
        base_rate=base_rate,
    )


def predict_p_tail(result: CalibrationResult, scores: np.ndarray) -> np.ndarray:
    """Apply the selected calibrator to new scores."""
    return _PREDICTORS[result.selected](result.calibrator, scores)
