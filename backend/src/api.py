# backend/src/api.py

import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .features import build_features

# Paths & model loading

# BASE_DIR points to backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

app = FastAPI(title="Stock Prediction API")

# CORS so React (localhost:3000 etc.) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    FEATURE_COLUMNS = model_bundle["feature_columns"]
except Exception as e:
    # If this fails, it's a hard error – better to crash early than return nonsense
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(symbol: str = Query(..., description="Stock ticker symbol, e.g. AAPL")):
    """
    Predict next-day direction (up/down) for the given stock symbol.
    """

    # Basic validation for symbol
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Query parameter 'symbol' is required")

    try:
        # Download recent data (last 2 years)
        end = datetime.today()
        start = end - timedelta(days=365 * 2)

        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data found for symbol '{symbol}'")

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Build features (inference mode: no target)
        feat_df = build_features(df, add_target=False)

        # Check that all expected feature columns exist
        missing = [c for c in FEATURE_COLUMNS if c not in feat_df.columns]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Missing expected feature columns: {missing}",
            )

        # Drop rows with NaNs in feature columns
        feat_df = feat_df.dropna(subset=FEATURE_COLUMNS)

        if feat_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough data to compute features for latest date",
            )

        # Take the last available row as the basis for prediction
        latest_row = feat_df.iloc[-1]
        X_latest = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)

        # Predict probability of going up
        prob_up = float(model.predict_proba(X_latest)[0, 1])
        direction = "up" if prob_up >= 0.5 else "down"

        return {
            "symbol": symbol,
            "as_of_date": latest_row.name.strftime("%Y-%m-%d"),
            "predicted_direction": direction,
            "prob_up": prob_up,
        }

    except HTTPException:
        # Re-raise clean HTTP errors we created above
        raise
    except Exception as e:
        # Catch any unexpected errors and return a safe 500
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing symbol '{symbol}': {e}",
        )
