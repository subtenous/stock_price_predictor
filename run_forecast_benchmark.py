import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# Random Forest
from sklearn.ensemble import RandomForestRegressor

# LSTM artifacts
import torch
import joblib
from learners.train_lstm import build_lstm_for_inference  # uses artifacts/lstm_config.pkl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Helpers: Metrics + Saving
# ----------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_scalar_float(x) -> float:
    """
    Safely convert pandas/numpy scalar-ish values to a Python float.
    """
    if isinstance(x, pd.Series):
        x = x.iloc[0]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[0, 0]
    if isinstance(x, np.generic):
        x = x.item()
    return float(x)

# ----------------------------
# Baseline: Naive Forecast
# ----------------------------
def naive_next_close(close_hist: pd.Series) -> float:
    # Predict tomorrow = today
    last_val = close_hist.iloc[-1]
    if isinstance(last_val, pd.Series):
        last_val = last_val.iloc[0]
    return float(last_val)


# ----------------------------
# ARIMA: One-step ahead
# ----------------------------
def arima_next_close(
    close_hist: pd.Series,
    cached_order: Optional[Tuple[int, int, int]] = None
) -> Tuple[float, Tuple[int, int, int]]:
    """
    Returns (y_hat, order_used).
    Uses auto_arima once (if order not cached) then re-fits ARIMA each step.
    """
    values = close_hist.astype(float).values

    if len(values) < 80:
        raise ValueError("Not enough history for ARIMA (need ~80+ points for stable fitting).")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if cached_order is None:
            model_auto = auto_arima(
                values,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
            cached_order = model_auto.order

        model = ARIMA(values, order=cached_order).fit()
        pred = model.forecast(steps=1)[0]

    return float(pred), cached_order


# ----------------------------
# RF: One-step ahead (close-only features)
# ----------------------------
RF_LAGS = [1, 2, 3, 5, 7, 14, 21, 30, 60]


def rf_make_training_table(close_hist: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close_hist.index)
    df["Target"] = close_hist.shift(-1)

    for lag in RF_LAGS:
        df[f"lag_{lag}"] = close_hist.shift(lag)

    df["ma_5"] = close_hist.rolling(5).mean()
    df["ma_20"] = close_hist.rolling(20).mean()

    delta = close_hist.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month

    return df.dropna()


def rf_build_latest_row(close_hist: pd.Series) -> pd.DataFrame:
    feats = {}

    def _as_float(x):
        if isinstance(x, pd.Series):
            x = x.iloc[0]
        if isinstance(x, np.generic):
            x = x.item()
        return float(x)

    for lag in RF_LAGS:
        if len(close_hist) <= lag:
            raise ValueError(f"Not enough history for RF lag {lag}")
        feats[f"lag_{lag}"] = _as_float(close_hist.iloc[-lag])

    feats["ma_5"] = float(close_hist.rolling(5).mean().iloc[-1])
    feats["ma_20"] = float(close_hist.rolling(20).mean().iloc[-1])

    delta = close_hist.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / (loss if loss != 0 else 1e-6)
    feats["rsi_14"] = float(100 - (100 / (1 + rs)))

    next_day = close_hist.index[-1] + pd.Timedelta(days=1)
    feats["day_of_week"] = int(next_day.dayofweek)
    feats["month"] = int(next_day.month)

    return pd.DataFrame([feats])


def rf_next_close(close_hist: pd.Series, n_estimators: int = 200) -> float:
    """
    Walk-forward friendly: fit RF on history available up to t, predict t+1.
    (This re-fits each step; we keep estimators modest so it's usable.)
    """
    if len(close_hist) < 150:
        raise ValueError("Not enough history for RF (need ~150+ points).")

    train_table = rf_make_training_table(close_hist)
    X = train_table.drop(columns=["Target"])
    y = train_table["Target"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    X_next = rf_build_latest_row(close_hist)
    X_next = X_next[X.columns]  # exact col order
    pred = model.predict(X_next)[0]
    return float(pred)


# ----------------------------
# LSTM: One-step ahead from artifacts
# ----------------------------
@dataclass
class LSTMArtifacts:
    model: torch.nn.Module
    lookback: int
    scaler: object  # sklearn scaler


def load_lstm_artifacts(artifacts_dir: str = "artifacts") -> LSTMArtifacts:
    model, lookback = build_lstm_for_inference()

    model_path = os.path.join(artifacts_dir, "lstm_model.pth")
    scaler_path = os.path.join(artifacts_dir, "lstm_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing {scaler_path}")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    scaler = joblib.load(scaler_path)

    return LSTMArtifacts(model=model, lookback=lookback, scaler=scaler)


def lstm_next_close(close_hist: pd.Series, artifacts: LSTMArtifacts) -> float:
    """
    Predict next close using:
    - artifacts.scaler fit on [Close, Close(t-1), ..., Close(t-lookback)]
    - model trained on scaled lags (shape: lookback x 1)
    """
    lb = artifacts.lookback

    if len(close_hist) < (lb + 1):
        raise ValueError(f"Not enough history for LSTM (need {lb+1}+ points).")

    last_vals = close_hist.astype(float).values[-(lb + 1):]  # len lb+1
    current = last_vals[-1]
    lags = last_vals[-2::-1]  # t-1 ... t-lb

    row = np.array([current] + list(lags), dtype=np.float32).reshape(1, -1)  # (1, lb+1)

    # Scale
    row_scaled = artifacts.scaler.transform(row)

    # X = lag columns, then flip to match training
    X_last = row_scaled[:, 1:]
    X_last = np.flip(X_last, axis=1)  # (1, lb)

    x = X_last.reshape(1, lb, 1).astype(np.float32)
    x_t = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        y_hat_scaled = float(artifacts.model(x_t).cpu().numpy().reshape(-1)[0])

    # Inverse transform: put prediction in column 0, dummy rest zeros
    dummy = np.zeros((1, lb + 1), dtype=np.float32)
    dummy[0, 0] = y_hat_scaled
    y_hat_price = artifacts.scaler.inverse_transform(dummy)[0, 0]

    return float(y_hat_price)


# ----------------------------
# Walk-forward benchmark
# ----------------------------
def download_close(symbol: str, start: str) -> pd.Series:
    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    close = df["Close"]

    # If yfinance gives a DataFrame, collapse to first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close = close.sort_index()
    return close


def walk_forward_one_step(
    close: pd.Series,
    test_steps: int,
    artifacts: Optional[LSTMArtifacts],
    do_lstm: bool,
    do_arima: bool,
    do_rf: bool,
    do_naive: bool,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For each step t in last test_steps:
      train = close[:t]
      actual = close[t+1]
      predict next-day close using each model
    """
    if len(close) < (test_steps + 100):
        raise ValueError("Not enough total history for requested test_steps.")

    # We need a next-day actual, so we test on indices where t+1 exists
    idx = close.index
    start_i = len(idx) - test_steps - 1
    end_i = len(idx) - 2

    rows = []
    arima_order = None

    for i in range(start_i, end_i + 1):
        t = idx[i]
        t_next = idx[i + 1]

        hist = close.loc[:t]
        actual = to_scalar_float(close.loc[t_next])

        rec = {
            "date": t_next.strftime("%Y-%m-%d"),
            "as_of": t.strftime("%Y-%m-%d"),
            "actual_close": actual,
        }

        if do_naive:
            rec["pred_naive"] = naive_next_close(hist)

        if do_lstm:
            rec["pred_lstm"] = lstm_next_close(hist, artifacts)

        if do_arima:
            pred, arima_order = arima_next_close(hist, cached_order=arima_order)
            rec["pred_arima"] = pred
            rec["arima_order"] = str(arima_order)

        if do_rf:
            rec["pred_rf"] = rf_next_close(hist, n_estimators=200)

        rows.append(rec)

    df_preds = pd.DataFrame(rows)

    # Metrics summary per model
    summary = {}
    for col in [c for c in df_preds.columns if c.startswith("pred_")]:
        name = col.replace("pred_", "")
        y_true = df_preds["actual_close"].values
        y_pred = df_preds[col].values
        summary[f"{name}_mae"] = float(mean_absolute_error(y_true, y_pred))
        summary[f"{name}_rmse"] = rmse(y_true, y_pred)

    return df_preds, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "AMZN", "TSLA", "NVDA"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--test-steps", type=int, default=60)
    parser.add_argument("--results-dir", default="results")

    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--no-arima", action="store_true")
    parser.add_argument("--no-rf", action="store_true")
    parser.add_argument("--no-naive", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.results_dir)

    do_lstm = not args.no_lstm
    do_arima = not args.no_arima
    do_rf = not args.no_rf
    do_naive = not args.no_naive

    lstm_artifacts = None
    if do_lstm:
        lstm_artifacts = load_lstm_artifacts("artifacts")

    all_summaries = []
    for symbol in args.tickers:
        print(f"\n=== Benchmarking {symbol} ===")
        close = download_close(symbol, start=args.start)

        df_preds, summary = walk_forward_one_step(
            close=close,
            test_steps=args.test_steps,
            artifacts=lstm_artifacts,
            do_lstm=do_lstm,
            do_arima=do_arima,
            do_rf=do_rf,
            do_naive=do_naive,
        )

        # Save per-ticker detailed predictions
        detail_path = os.path.join(args.results_dir, f"{symbol}_walkforward_{args.test_steps}.csv")
        df_preds.to_csv(detail_path, index=False)
        print(f"Saved: {detail_path}")

        # Save summary row
        row = {"symbol": symbol, "test_steps": args.test_steps}
        row.update(summary)
        all_summaries.append(row)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(args.results_dir, "benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Done ===")
    print(summary_df)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()