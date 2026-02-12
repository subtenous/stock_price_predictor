from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from forecast_lstm_live import recursive_forecast_lstm
from learners.train_arima import arima_forecast_next_days
from learners.train_random_forest import rf_forecast_next_days
import traceback
from fastapi import HTTPException
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
def forecast(
    symbol: str = Query(...),
    days: int = Query(30, ge=1, le=60),
    model: str = Query("lstm", pattern="^(lstm|arima|rf)$"),
):
    try:
        if model == "lstm":
            return recursive_forecast_lstm(symbol, days)

        if model == "arima":
            return arima_forecast_next_days(symbol, days)

        if model == "rf":
            try:
                return rf_forecast_next_days(symbol, days)
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

    except FileNotFoundError as e:
        # Missing model weights, scaler, etc
        raise HTTPException(status_code=500, detail=f"Server configuration error: {str(e)}")

    except ValueError as e:
        msg = str(e).lower()

        if "no data returned" in msg or "no data found" in msg:
            raise HTTPException(status_code=404, detail=str(e))

        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected crash
        raise HTTPException(status_code=500, detail="Internal server error")
