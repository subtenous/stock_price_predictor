import numpy as np
import pandas as pd
import yfinance as yf
import torch
import joblib

from learners.train_lstm import build_lstm_for_inference  # adjust path if needed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recursive_forecast_lstm(symbol: str, days: int = 30, start="2010-01-01"):
    # ---- Load model + LSTM-specific artifacts ----
    model, lookback = build_lstm_for_inference()
    model.load_state_dict(torch.load("artifacts/lstm_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    lstm_scaler = joblib.load("artifacts/lstm_scaler.pkl")

    # Debug prints (optional but useful)
    # print("LSTM lookback:", lookback)
    # print("Scaler n_features_in_:", getattr(lstm_scaler, "n_features_in_", None))

    # ---- Fetch latest data ----
    df = yf.download(symbol, start=start, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    close = df["Close"].astype(float).values

    # Need at least lookback+1 closes to create [current + 7 lags]
    if len(close) < (lookback + 1):
        raise ValueError(f"Not enough data: need at least {lookback+1} closes, got {len(close)}")

    # ---- Build the last feature row EXACTLY like training ----
    # Training scaler was fit on shift_df values with columns:
    # [Close_scaled, Close_scaled(t-1), ..., Close_scaled(t-7)]
    #
    # In training, Close_scaled already existed, but then you fit a new scaler
    # on this 8-column matrix. For live inference we will feed raw closes into
    # the same 8-column structure, then use lstm_scaler.transform().
    #
    # Construct row: [current_close, lag1_close, ..., lag7_close]
    last_8 = close[-(lookback + 1):]  # length 8
    current = last_8[-1]
    lags = last_8[-2::-1]  # previous values reversed: t-1, t-2, ..., t-7
    row = np.array([current] + list(lags), dtype=np.float32).reshape(1, -1)  # shape (1, 8)

    # Scale with the saved LSTM scaler
    row_scaled = lstm_scaler.transform(row)  # shape (1, 8)

    # X is columns 1..end, then flipped to match training
    X_last = row_scaled[:, 1:]
    X_last = np.flip(X_last, axis=1)  # shape (1, 7)

    # This is our rolling window in scaled-space that the model expects
    window = X_last.flatten().astype(np.float32).tolist()  # length 7

    preds_scaled = []

    # ---- Recursive multi-step forecast (scaled space) ----
    for _ in range(days):
        x = np.array(window, dtype=np.float32).reshape(1, lookback, 1)
        x_t = torch.from_numpy(x).to(DEVICE)

        with torch.no_grad():
            y_hat = model(x_t).cpu().numpy().reshape(-1)[0]  # scaled prediction

        preds_scaled.append(float(y_hat))

        # Update window: drop oldest, append new prediction
        window.pop(0)
        window.append(float(y_hat))

    # ---- Inverse transform back to price units ----
    # scaler expects 8 columns; prediction belongs in column 0
    dummy = np.zeros((days, lookback + 1), dtype=np.float32)  # (days, 8)
    dummy[:, 0] = np.array(preds_scaled, dtype=np.float32)
    preds_price = lstm_scaler.inverse_transform(dummy)[:, 0]

    # Future dates (business days)
    last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)

    return {
        "symbol": symbol.upper(),
        "as_of": last_date.strftime("%Y-%m-%d"),
        "horizon_days": days,
        "predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": float(p)}
            for d, p in zip(future_dates, preds_price)
        ],
    }


if __name__ == "__main__":
    print(recursive_forecast_lstm("AAPL", 30))
