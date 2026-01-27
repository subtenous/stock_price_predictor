# learners/forecast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import numpy as np
import pandas as pd
import torch

@dataclass
class ForecastResult:
    dates: List[str]
    predicted_close: List[float]          # unscaled
    predicted_close_scaled: List[float]   # scaled (optional)

def _make_future_dates(last_date: pd.Timestamp, steps: int) -> List[str]:
    # Business days (markets) is usually the right default
    idx = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=steps)
    return [d.strftime("%Y-%m-%d") for d in idx]

def _recursive_forecast_scaled(
    model: torch.nn.Module,
    last_window_scaled: np.ndarray,
    steps: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generic recursive forecaster for models that predict the next Close_scaled
    given a window of previous Close_scaled values.

    last_window_scaled: shape (lookback,)
    returns: shape (steps,)
    """
    model.eval()
    window = last_window_scaled.astype(np.float32).copy().tolist()
    preds = []

    for _ in range(steps):
        x = np.array(window, dtype=np.float32)[None, :, None]  # (1, lookback, 1)
        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            y_hat = model(x_t)  # expected (1,1) or (1,) or (1, something)
        y_hat = y_hat.detach().cpu().numpy().reshape(-1)[0]

        preds.append(float(y_hat))
        window.pop(0)
        window.append(float(y_hat))

    return np.array(preds, dtype=np.float32)

def forecast_single_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    scaler,
    lookback: int,
    steps: int = 30,
    device: str = "cpu",
) -> ForecastResult:
    """
    df must contain a 'Close_scaled' column.
    scaler must support inverse_transform (like sklearn MinMaxScaler/StandardScaler).
    """
    if "Close_scaled" not in df.columns:
        raise ValueError("DataFrame must include 'Close_scaled' column")

    if len(df) < lookback:
        raise ValueError(f"Not enough data to forecast. Need >= {lookback} rows.")

    last_window = df["Close_scaled"].values[-lookback:]
    preds_scaled = _recursive_forecast_scaled(model, last_window, steps=steps, device=device)

    # inverse scale back to price
    preds_unscaled = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    last_date = pd.to_datetime(df.index[-1])
    future_dates = _make_future_dates(last_date, steps)

    return ForecastResult(
        dates=future_dates,
        predicted_close=[float(x) for x in preds_unscaled],
        predicted_close_scaled=[float(x) for x in preds_scaled],
    )

def forecast_ensemble_mean(
    forecasts: Dict[str, ForecastResult]
) -> ForecastResult:
    """
    Simple mean ensemble over predicted_close (unscaled).
    Assumes all models produced the same dates list.
    """
    keys = list(forecasts.keys())
    if not keys:
        raise ValueError("No forecasts to ensemble")

    dates = forecasts[keys[0]].dates
    mat = np.vstack([forecasts[k].predicted_close for k in keys])  # (n_models, steps)
    mean_preds = mat.mean(axis=0)

    # keep scaled empty since we ensemble unscaled here
    return ForecastResult(
        dates=dates,
        predicted_close=[float(x) for x in mean_preds],
        predicted_close_scaled=[],
    )
