# src/features.py
import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(df: pd.DataFrame, add_target:bool = True) -> pd.DataFrame:
    """
    df: DataFrame with at least ['Open', 'High', 'Low', 'Close', 'Volume']
    index: DatetimeIndex
    Returns: df with added feature columns and a 'target' column (if possible)
    """

    df = df.copy()

    # Just in case, ensure 'Close' and 'Volume' are Series, not DataFrames
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]
    if isinstance(df["Volume"], pd.DataFrame):
        df["Volume"] = df["Volume"].iloc[:, 0]

    # Basic returns (explicitly disable fill_method to silence FutureWarning)
    df["return_1d"] = df["Close"].pct_change(fill_method=None)
    df["return_5d"] = df["Close"].pct_change(5, fill_method=None)
    df["return_10d"] = df["Close"].pct_change(10, fill_method=None)

    # Moving averages
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["ma_20"] = df["Close"].rolling(window=20).mean()

    # Volatility (rolling std of returns)
    df["vol_5"] = df["return_1d"].rolling(window=5).std()
    df["vol_10"] = df["return_1d"].rolling(window=10).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["Close"], period=14)

    # Volume features
    df["vol_zscore_20"] = (
        (df["Volume"] - df["Volume"].rolling(window=20).mean())
        / df["Volume"].rolling(window=20).std()
    )

    if add_target:
        # Target: next-day direction (1 = up, 0 = down or equal)
        df["close_tomorrow"] = df["Close"].shift(-1)
        df["target"] = (df["close_tomorrow"] > df["Close"]).astype(int)

    return df

    