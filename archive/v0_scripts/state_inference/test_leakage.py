"""
leakage tests to test shuffling yields random and look ahead improves
"""
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


FEATURE_COLS = [
      "event_count_500ms",
      "buy_count_500ms",
      "sell_count_500ms",
      "size_sum_500ms",
      "size_mean_500ms",
      "buy_imbalance_500ms",
]



def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

def test_shuffle(supervised_df):
    X = supervised_df.select(FEATURE_COLS).to_numpy()
    y = supervised_df["regime_id"].to_numpy()
    pre_acc = train_and_eval(X,y)

    y = np.random.permutation(y)
    post_acc = train_and_eval(X, y)

    print(f" Baseline Accuracy: {pre_acc}")
    print(f" Post-Shuffle Accuracy: {post_acc}")

def test_future_shift(supervised_df):
    X = supervised_df.select(FEATURE_COLS).to_numpy()
    y = supervised_df["regime_id"].to_numpy()
    pre_acc = train_and_eval(X,y)

    shifted_df = supervised_df.with_columns([
        pl.col(c).shift(-1) for c in FEATURE_COLS
    ]).drop_nulls()
    
    X = shifted_df.select(FEATURE_COLS).to_numpy()
    y = shifted_df["regime_id"].to_numpy()
    post_acc = train_and_eval(X,y)

    print(f" Baseline Accuracy: {pre_acc}")
    print(f" Post-Shift Accuracy: {post_acc}")

    

def main():
    supervised = pl.read_parquet("data/processed/supervised.parquet")

    print("LEAKAGE TESTS")
    test_shuffle(supervised)
    test_future_shift(supervised)


if __name__ == "__main__":
    main()