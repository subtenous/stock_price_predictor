# backend/src/predict_script.py

import os
import joblib
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

from .features import build_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


def predict_ticker(symbol: str = "AAPL"):
    print(f"Loading model from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    # Download recent data
    end = datetime.today()
    start = end - timedelta(days=365)

    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"No data found for ticker: {symbol}")
        return

    # 🔹 Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build features (inference mode: no target)
    feat_df = build_features(df, add_target=False)

    # 🔹 Now it's safe to drop NaNs on the known feature columns
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print("Feature columns missing from feat_df:", missing)
        print("Available columns:", list(feat_df.columns))
        return

    feat_df = feat_df.dropna(subset=feature_cols)

    if feat_df.empty:
        print("Not enough data to compute features after dropna.")
        return

    latest_row = feat_df.iloc[-1]
    X = latest_row[feature_cols].values.reshape(1, -1)

    prob_up = float(model.predict_proba(X)[0, 1])
    direction = "UP" if prob_up >= 0.5 else "DOWN"

    print("\n--- Prediction Result ---")
    print(f"Ticker: {symbol}")
    print(f"Predicted direction: {direction}")
    print(f"Probability of going up: {prob_up:.4f}")


if __name__ == "__main__":
    predict_ticker("AAPL")
