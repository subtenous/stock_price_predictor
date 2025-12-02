# src/train_model.py

import os
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from .features import build_features

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "vol_5",
    "vol_10",
    "rsi_14",
    "vol_zscore_20",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../backend
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

def download_data(tickers: List[str],
                  start: str = "2015-01-01",
                  end: str = None) -> pd.DataFrame:
    """
    Download daily OHLCV for multiple tickers and concatenate with 'Ticker' column.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    frames = []
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,   # be explicit
            progress=False
        )
        if df.empty:
            print(f"[WARN] No data for {ticker}")
            continue

        # If yfinance returns a MultiIndex on columns, flatten it
        if isinstance(df.columns, pd.MultiIndex):
            # keep only the first level: Open, High, Low, Close, etc.
            df.columns = df.columns.get_level_values(0)

        df["Ticker"] = ticker
        frames.append(df)

    if not frames:
        raise ValueError("No data downloaded for any ticker")

    all_data = pd.concat(frames)
    return all_data

def prepare_dataset(tickers: List[str]) -> pd.DataFrame:
    df = download_data(tickers)
    # Group by Ticker and build features per stock
    all_feat = []
    for ticker, g in df.groupby("Ticker"):
        g_feat = build_features(g, add_target=True)
        all_feat.append(g_feat)

    feat_df = pd.concat(all_feat)
    # Drop rows with NaNs from rolling/RSI/target shift
    feat_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["target"])
    return feat_df

def train_and_save_model(tickers: List[str], model_path: str = MODEL_PATH):
    feat_df = prepare_dataset(tickers)

    # Time-based split: use date index
    feat_df = feat_df.sort_index()
    # 70% train, 30% test
    split_idx = int(len(feat_df) * 0.7)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["target"].values

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["target"].values

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # Simple RF baseline
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )
    print(f"✅ Saved model to {os.path.abspath(model_path)}")

if __name__ == "__main__":
    # Start with a small universe
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    train_and_save_model(TICKERS)