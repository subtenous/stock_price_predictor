from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from forecast_lstm_live import recursive_forecast_lstm

app = FastAPI(title="Live Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast")
def forecast(symbol: str = Query(...), days: int = Query(30, ge=1, le=60)):
    return recursive_forecast_lstm(symbol, days)
