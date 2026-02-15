from pathlib import Path

from ipo_risk_engine.torch.data import normalize_features
from ipo_risk_engine.torch.data import SupervisedDataset
from ipo_risk_engine.torch.models import MLPClassifier
from ipo_risk_engine.torch.train import train_one_epoch
from ipo_risk_engine.torch.train import evaluate
from scripts.state_inference.test_leakage import FEATURE_COLS

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import torch
import polars as pl
import numpy as np


def main():
    supervised = pl.read_parquet("data/processed/supervised.parquet")
    X = supervised.select(FEATURE_COLS).to_numpy()
    y = supervised["regime_id"].to_numpy()

    X_norm, mean, std = normalize_features(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y, test_size=0.2, random_state=42)

    training_set = SupervisedDataset(X_train, y_train)
    training_loader = DataLoader(training_set, shuffle=True, batch_size=64)

    val_set = SupervisedDataset(X_val, y_val)
    val_loader = DataLoader(val_set, batch_size=64)

    model = MLPClassifier(input_dim=X.shape[1], hidden_dim=32, num_classes=len(np.unique(y)))
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = 100
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, training_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if (epoch + 1) % 10 == 0:
              print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f},val_acc={val_acc:.4f}")

      # Save checkpoint
    Path("artifacts/runs").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "artifacts/runs/mlp_baseline.pt")
    print(f"Model saved. Final val_acc: {val_acc:.4f}")




if __name__ == "__main__":
    main()