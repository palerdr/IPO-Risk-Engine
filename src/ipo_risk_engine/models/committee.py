"""
RiverCommittee: Ridge + KNN ensemble for RIVER street risk_severity prediction.

Design:
  - Both models use StandardScaler
  - dollar_volume_mean gets log1p transform before scaling
  - predict() returns the average of both model predictions
  - predict_individual() returns both for disagreement-as-uncertainty
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class RiverCommittee:
    """Two-model committee for RIVER street risk severity prediction."""

    def __init__(
        self,
        ridge_alpha: float = 1000.0,
        knn_k: int = 7,
        dollar_volume_col: int | None = None,
    ):
        """
        Args:
            ridge_alpha: Ridge regularization strength.
            knn_k: Number of neighbors for KNN.
            dollar_volume_col: Column index of dollar_volume_mean for log transform.
                If None, no log transform is applied.
        """
        self.ridge_alpha = ridge_alpha
        self.knn_k = knn_k
        self.dollar_volume_col = dollar_volume_col

        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=ridge_alpha)
        self.knn = KNeighborsRegressor(n_neighbors=knn_k)
        self._is_fitted = False

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Log-transform dollar_volume + scale. Copies to avoid mutating caller's array."""
        X = X.copy()
        if self.dollar_volume_col is not None:
            X[:, self.dollar_volume_col] = np.log1p(X[:, self.dollar_volume_col])
        return self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RiverCommittee:
        X = self._transform(X, fit=True)
        self.ridge.fit(X, y)
        self.knn.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._transform(X)
        return np.mean([self.ridge.predict(X), self.knn.predict(X)], axis=0)

    def predict_individual(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._transform(X)
        return self.ridge.predict(X), self.knn.predict(X)



def prepare_features(
    df,
    feature_cols: list[str],
) -> tuple[np.ndarray, int | None]:
    """Extract feature matrix from DataFrame with dollar_volume_mean index detection.

    Returns:
        (X, dv_idx) where dv_idx is the column index of dollar_volume_mean, or None.
    """
    X = df.select(feature_cols).to_numpy().astype(float)
    dv_idx = feature_cols.index("dollar_volume_mean") if "dollar_volume_mean" in feature_cols else None
    return X, dv_idx
