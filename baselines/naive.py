# baselines/naive.py
import pandas as pd
import yfinance as yf


def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a 1D Close series no matter what yfinance gives back.
    """
    close = df["Close"]

    # If it's a DataFrame (e.g., MultiIndex columns or weird shape), take the first column
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce").dropna()
    return close


def naive_forecast_next_days(symbol: str, days: int = 30, start: str = "2010-01-01"):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required")

    df = yf.download(symbol, start=start, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'")

    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance data did not include 'Close' column")

    close = _get_close_series(df)
    if close.empty:
        raise ValueError(f"No usable close prices for symbol '{symbol}'")

    last_close = float(close.iloc[-1])
    last_date = pd.to_datetime(close.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)

    return {
        "symbol": symbol,
        "as_of": last_date.strftime("%Y-%m-%d"),
        "horizon_days": days,
        "predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": last_close}
            for d in future_dates
        ],
        "model_info": {"name": "naive_persistence", "rule": "predict_t+1 = close_t"},
    }
