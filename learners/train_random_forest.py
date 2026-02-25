from pyexpat import features
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os


class RandomForestModel:
    def __init__(self, split_ratio=0.80, skip_ratio=0.10, EXTERNAL_TICKERS=None, TICKER=None):
        print(EXTERNAL_TICKERS, TICKER)
        self.config = {
            "TICKER": TICKER,
            "START_DATE": "2017-01-01",
            "END_DATE": "2022-12-31",
            "LOOKBACK_LAGS": [1, 2, 3, 5, 7, 14, 21, 30, 60],
            "TRAIN_TEST_SPLIT_RATIO": split_ratio,
            "EXTERNAL_TICKERS": EXTERNAL_TICKERS,
            "RF_PARAMS": {
                "n_estimators": 500,
                "max_depth": 20,
                "random_state": 42,
                "n_jobs": -1,
            },
            "SKIP_RATIO": skip_ratio,
        }
        self.model = None
        self.feature_names = None

    def fetch_and_merge_data(self):
        print("Fetching and merging data for Random Forest...")
        # Fetch main ticker data
        main_data = yf.download(
            self.config["TICKER"],
            start=self.config["START_DATE"],
            end=self.config["END_DATE"],
        )
        main_data = main_data[["Open", "High", "Low", "Close", "Volume"]].add_prefix(
            f"{self.config['TICKER']}_"
        )

        # Fetch external ticker data
        all_data_frames = [main_data]
        for ticker in self.config["EXTERNAL_TICKERS"]:
            try:
                ext_data = yf.download(
                    ticker, start=self.config["START_DATE"], end=self.config["END_DATE"]
                )
                ext_data = ext_data[["Close"]].rename(
                    columns={"Close": f"{ticker}_Close"}
                )
                all_data_frames.append(ext_data)
                print(f"✓ Fetched {ticker}")
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {e}")

        # Combine all data
        combined = pd.concat(all_data_frames, axis=1)
        return combined.ffill().dropna()

    def create_features(self, data):
        print("Creating features for Random Forest...")
        features = pd.DataFrame(index=data.index)
        main_close_col = f"{self.config['TICKER']}_Close"

        # 1. Target Variable (next day's close price)
        features["Target"] = data[main_close_col].shift(-1)

        # 2. Lag Features for main ticker
        for lag in self.config["LOOKBACK_LAGS"]:
            features[f"lag_{lag}"] = data[main_close_col].shift(lag)

        # 3. Technical Indicators for main ticker
        features["ma_5"] = data[main_close_col].rolling(5).mean()
        features["ma_20"] = data[main_close_col].rolling(20).mean()

        delta = data[main_close_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-6)
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # 4. Calendar Features
        features["day_of_week"] = data.index.dayofweek
        features["month"] = data.index.month

        # 5. External Ticker Features
        for ticker in self.config["EXTERNAL_TICKERS"]:
            col_name = f"{ticker}_Close"
            if col_name in data.columns:
                features[f"{ticker}_close_scaled"] = (
                    data[col_name] / data[col_name].iloc[0]
                )

        return features.dropna()

    def build_latest_feature_row_from_close(self, close_series) -> pd.DataFrame:
    
        # ---- Normalize input to a 1D float Series ----
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        close_series = pd.Series(close_series).astype(float)

        if not isinstance(close_series.index, pd.DatetimeIndex):
            # fallback: assume last day is "today"
            close_series.index = pd.date_range(end=pd.Timestamp.today(), periods=len(close_series), freq="B")

        # ---- Helper to ensure scalar extraction ----
        def _scalar(x):
            # handles numpy scalars cleanly
            return float(np.asarray(x).reshape(-1)[0])

        features = {}

        # ---- Lags ----
        for lag in self.config["LOOKBACK_LAGS"]:
            if len(close_series) <= lag:
                raise ValueError(f"Not enough history for lag {lag} (need > {lag}, got {len(close_series)})")
            features[f"lag_{lag}"] = _scalar(close_series.iloc[-lag])

        # ---- Moving averages ----
        features["ma_5"] = _scalar(close_series.rolling(5).mean().iloc[-1])
        features["ma_20"] = _scalar(close_series.rolling(20).mean().iloc[-1])

        # ---- RSI 14 ----
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = _scalar(gain) / (_scalar(loss) if _scalar(loss) != 0 else 1e-6)
        features["rsi_14"] = float(100 - (100 / (1 + rs)))

        # ---- Calendar features (next business day) ----
        last_dt = close_series.index[-1]
        next_day = (last_dt + pd.offsets.BDay(1))
        features["day_of_week"] = int(next_day.dayofweek)
        features["month"] = int(next_day.month)

        return pd.DataFrame([features])

        

    def prepare_data(self, featured_data):
        print("Preparing train/skip/test split...")
        X = featured_data.drop("Target", axis=1)
        y = featured_data["Target"]
        self.feature_names = X.columns.tolist()

        n = len(featured_data)
        train_end = int(n * self.config["TRAIN_TEST_SPLIT_RATIO"])   # 0.8n
        skip_size = int(n * self.config["SKIP_RATIO"])               # 0.1n
        test_start = train_end + skip_size                           # 0.9n

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test   = X[test_start:], y[test_start:]  # last 10%

        print(f"Train set size: {len(X_train)}, Skip size: {skip_size}, Test set size: {len(X_test)}")
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(**self.config["RF_PARAMS"])
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def predict(self, X_test):
        print("Generating predictions...")
        return self.model.predict(X_test)

    def save_results(self, predictions, y_test):
        if not os.path.exists("predictions"):
            os.makedirs("predictions")

        np.save("predictions/rf_predictions.npy", predictions)
        print("Random Forest predictions saved to 'predictions/rf_predictions.npy'")

        # Ensure actuals are saved for the same test period
        if not os.path.exists("predictions/actuals.npy"):
            np.save("predictions/actuals.npy", y_test.values)
            print("Actual values for the test set saved to 'predictions/actuals.npy'")


def train_random_forest_model(split_ratio, skip_ratio, EXTERNAL_TICKERS=None, TICKER=None):
    """
    Main function to orchestrate the Random Forest model training and prediction process.
    This function will be called by your start.py script.
    """
    print("--- Starting Random Forest Model Training and Prediction ---")
    print(f"Using external tickers: {EXTERNAL_TICKERS}, Target ticker: {TICKER}")

    model_handler = RandomForestModel(split_ratio, skip_ratio, EXTERNAL_TICKERS, TICKER)

    # 1. Get and process data
    combined_data = model_handler.fetch_and_merge_data()
    featured_data = model_handler.create_features(combined_data)

    # 2. Split data and train
    X_train, y_train, X_test, y_test = model_handler.prepare_data(featured_data)
    model_handler.train_model(X_train, y_train)

    # 3. Predict and save
    predictions = model_handler.predict(X_test)
    model_handler.save_results(predictions, y_test)

    print("--- Random Forest Model Training and Prediction Complete ---")

    # The main system expects the model object and the raw prediction values
    return model_handler.model, predictions


def rf_forecast_next_days(symbol: str, days: int = 30, start: str = "2010-01-01"):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("Symbol is required")

    df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for symbol '{symbol}'")

    if "Close" not in df.columns:
        raise ValueError("Yahoo Finance response missing 'Close' column")

    close = df["Close"].astype(float).dropna()
    if len(close) < 250:
        raise ValueError("Not enough history to train RF reliably (need ~250+ points)")

    handler = RandomForestModel(
        split_ratio=0.8,
        skip_ratio=0.1,
        EXTERNAL_TICKERS=[],
        TICKER=symbol
    )

    featured = pd.DataFrame(index=close.index)
    featured["Target"] = close.shift(-1)

    for lag in handler.config["LOOKBACK_LAGS"]:
        featured[f"lag_{lag}"] = close.shift(lag)

    featured["ma_5"] = close.rolling(5).mean()
    featured["ma_20"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-6)
    featured["rsi_14"] = 100 - (100 / (1 + rs))

    featured["day_of_week"] = close.index.dayofweek
    featured["month"] = close.index.month

    featured = featured.dropna()
    if len(featured) < 100:
        raise ValueError("Not enough usable rows after feature engineering (lags/rolling windows removed too much data)")

    X = featured.drop("Target", axis=1)
    y = featured["Target"]

    model = RandomForestRegressor(**handler.config["RF_PARAMS"])
    model.fit(X, y)

    handler.model = model
    handler.feature_names = X.columns.tolist()

    preds = []
    sim_close = close.copy()

    for _ in range(days):
        X_next = handler.build_latest_feature_row_from_close(sim_close)
        X_next = X_next[handler.feature_names]
        y_hat = float(handler.model.predict(X_next)[0])
        preds.append(y_hat)

        next_date = pd.bdate_range(sim_close.index[-1] + pd.Timedelta(days=1), periods=1)[0]
        sim_close.loc[next_date] = y_hat

    last_date = pd.to_datetime(close.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)

    return {
        "symbol": symbol,
        "as_of": last_date.strftime("%Y-%m-%d"),
        "horizon_days": days,
        "predictions": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": float(p)}
            for d, p in zip(future_dates, preds)
        ],
        "model_info": {"n_obs": int(len(close)), "n_train_rows": int(len(featured))},
    }
    

def plot_results(y_test, y_pred, ticker):
    """Helper function for visualizing results when run standalone."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test, label="Actual Price", color="blue", alpha=0.8)
    plt.plot(
        y_test.index, y_pred, label="Predicted Price", color="orange", linestyle="--"
    )
    plt.title(f"{ticker} Price Predictions (Random Forest)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # This block allows you to run the script by itself for testing
    model_obj, preds = train_random_forest_model()

    # To plot, we need to re-run parts of the logic to get y_test
    # This is just for standalone testing convenience
    test_handler = RandomForestModel()
    test_combined = test_handler.fetch_and_merge_data()
    test_featured = test_handler.create_features(test_combined)
    _, _, _, y_test = test_handler.prepare_data(test_featured)

    plot_results(y_test, preds, test_handler.config["TICKER"])
