import os
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import joblib

from learners.train_lstm import build_lstm_for_inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recursive_forecast_lstm(symbol: str, days: int = 30, start: str = "2010-01-01"):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required")

    # ---- Load artifacts ----
    artifacts_dir = "artifacts"
    model_path = os.path.join(artifacts_dir, "lstm_model.pth")
    scaler_path = os.path.join(artifacts_dir, "lstm_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model weights: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    model, lookback = build_lstm_for_inference()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    lstm_scaler = joblib.load(scaler_path)

    # ---- Fetch latest data ----
    df = yf.download(symbol, start=start, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance data did not include 'Close' column")

    close = df["Close"].astype(float).dropna().values
    if len(close) < (lookback + 1):
        raise ValueError(
            f"Not enough data: need at least {lookback+1} closes, got {len(close)}"
        )

    # ---- Build input window (must match training pipeline) ----
    last_vals = close[-(lookback + 1):]  # e.g. 8 values for lookback=7
    current = last_vals[-1]
    lags = last_vals[-2::-1]  # t-1 ... t-lookback

    row = np.array([current] + list(lags), dtype=np.float32).reshape(1, -1)

    #print("Scaler features:", getattr(lstm_scaler, "n_features_in_", None))
    #print("Row min/max:", row.min(), row.max())
    #test_scaled = lstm_scaler.transform(row)
    #print("Scaled row min/max:", test_scaled.min(), test_scaled.max())

    
    # NOTE: This assumes lstm_scaler was fit on values in same "space" as row.
    row_scaled = lstm_scaler.transform(row)

    X_last = row_scaled[:, 1:]
    X_last = np.flip(X_last, axis=1)
    window = X_last.flatten().astype(np.float32).tolist()

    preds_scaled = []

    # ---- Recursive forecast ----
    for _ in range(days):
        x = np.array(window, dtype=np.float32).reshape(1, lookback, 1)
        x_t = torch.from_numpy(x).to(DEVICE)

        with torch.no_grad():
            y_hat = float(model(x_t).cpu().numpy().reshape(-1)[0])

        preds_scaled.append(y_hat)
        window.pop(0)
        window.append(y_hat)

    # ---- Inverse back to price units ----
    dummy = np.zeros((days, lookback + 1), dtype=np.float32)
    dummy[:, 0] = np.array(preds_scaled, dtype=np.float32)
    preds_price = lstm_scaler.inverse_transform(dummy)[:, 0]

    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)

    return {
        "symbol": symbol,
        "as_of": last_date.strftime("%Y-%m-%d"),
        "horizon_days": days,
        "predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(float(p), 2)}
            for d, p in zip(future_dates, preds_price)
        ],
    }

    

if __name__ == "__main__":
    print(recursive_forecast_lstm("AAPL", 30))
