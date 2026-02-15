import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset

def normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_norm = (X-means)/(stds + 1e-8)
    return X_norm, means, stds

class SupervisedDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])
