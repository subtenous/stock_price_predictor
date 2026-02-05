from fastapi import FastAPI, Query, HTTPException
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
    try:
        return recursive_forecast_lstm(symbol, days)

    # ValueError for user issues (invalid ticker / not enough data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Missing trained artifacts is a server/config issue
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Catch-all so the server never crashes
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error while generating forecast")
