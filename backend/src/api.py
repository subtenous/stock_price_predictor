# src/api.py
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .features import build_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")

app = FastAPI(title="Stock Prediction API")

# CORS so your React dev server can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
FEATURE_COLUMNS = model_bundle["feature_columns"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(symbol: str):
    # Download recent data (e.g. last 1–2 years, to ensure indicators have enough history)
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
        raise HTTPException(status_code=400, detail=f"No data for symbol {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    feat_df = build_features(df, add_target=False)
    feat_df = feat_df.dropna(subset=FEATURE_COLUMNS)

    if feat_df.empty:
        raise HTTPException(status_code=500, detail="Not enough data to build features")

    # Take the last available row
    latest_row = feat_df.iloc[-1]
    X_latest = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)

    # Predict
    prob_up = float(model.predict_proba(X_latest)[0, 1])
    direction = "up" if prob_up >= 0.5 else "down"

    return {
        "symbol": symbol,
        "as_of_date": latest_row.name.strftime("%Y-%m-%d"),
        "predicted_direction": direction,
        "prob_up": prob_up,
    }