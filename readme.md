# Stock Price Prediction System

**Final Year Project - Computer Science**  
**University of Portsmouth**  
**Author:** Marcus Cameron  

## Project Overview

This repository contains the implementation for my Final Year Project, titled **Design and Evaluation of a Stock Price Prediction System Using Machine Learning Models**.

The project focuses on integrating and evaluating stock price forecasting models within a practical software system. It includes a Python/FastAPI backend for retrieving stock data and generating forecasts, and a React frontend prototype for selecting forecasting options and visualising returned predictions.

The system was developed as an engineering project to explore the gap between research-based forecasting models and practical user-facing applications.

## Background

Financial markets are complex, volatile, and difficult to predict reliably. While many statistical, machine learning, and deep learning models have been proposed in academic literature, fewer are integrated into accessible systems that allow users to interact with live data and view predictions through an interface.

This project addresses that gap by combining forecasting models, a backend API, and a frontend prototype into a functional end-to-end system.

## Attribution / Starting Point

This work was initially informed by and adapted from the following open-source research repository:

[https://github.com/micfun123/AI-stock-prediction](https://github.com/micfun123/AI-stock-prediction)

The original repository provided a research-based stock prediction framework and examples of model architectures. The adaptations, backend API implementation, frontend prototype, evaluation work, and system integration were completed as part of this project.

## My Main Contributions
- Adapted the existing research codebase for live forecasting through a FastAPI backend.
- Added new live forecasting functions for ARIMA and Random Forest to support API-based predictions.
- Implemented recursive multi-step forecasting so the system can generate configurable future trading-day forecasts.
- Added JSON response formatting so predictions can be consumed directly by the React frontend.
- Implemented a React frontend prototype for ticker, model, and forecast horizon selection.
- Added evaluation and testing across multiple tickers and forecasting models.

## Main Features

- Retrieves historical stock data using `yfinance`
- Supports multiple forecasting approaches
- Includes Naive baseline, ARIMA, Random Forest, and exploratory LSTM work
- Provides forecasts through a FastAPI REST API
- Supports configurable stock ticker, model type, and forecast horizon
- Returns predictions in structured JSON format
- Includes a React/Vite frontend prototype
- Displays forecasts using a line chart and table
- Includes basic loading and error handling in the frontend
- Supports short and longer trading-day forecast horizons

## Repository Structure

```text
stock_price_predictor/
├── api_live.py                 # FastAPI backend exposing the /forecast endpoint
├── start.py                    # Original orchestration script for model training/evaluation
├── forecast_lstm_live.py        # LSTM forecasting logic used during development
├── learners/                   # Individual model implementations
├── meta_learner/               # Meta-learner files from the adapted research implementation
├── utils/                      # Data loading and preprocessing utilities
├── baselines/                  # Baseline model-related files
├── results/                    # Evaluation outputs and results
├── frontend/                   # React/Vite frontend prototype
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files and folders
└── README.md                   # Project documentation
```

## Backend Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI backend:

```bash
uvicorn api_live:app --reload
```

The backend should run locally at:

```text
http://127.0.0.1:8000
```

## API Usage

The main endpoint is:

```http
GET /forecast
```

Example request:

```text
http://127.0.0.1:8000/forecast?symbol=AAPL&days=5&model=arima
```

Example parameters:

| Parameter | Description | Example |
|---|---|---|
| `symbol` | Stock ticker symbol | `AAPL` |
| `days` | Forecast horizon in trading days | `5` |
| `model` | Forecasting model | `arima` |

Example response:

```json
{
  "symbol": "AAPL",
  "as_of": "2026-05-01",
  "horizon_days": 5,
  "predictions": [
    {
      "date": "2026-05-04",
      "predicted_close": 280.13
    }
  ],
  "model_info": {
    "order": [3, 1, 5],
    "n_obs": 4107
  }
}
```

## Forecasting Approach

The system supports recursive multi-step forecasting. In this approach, the model predicts the next trading day, then that prediction is fed back into the model to generate later predictions.

This allows the system to produce multi-day forecasts, such as 5, 10, 20, or 30 trading days. However, longer forecasts should be interpreted with caution because recursive forecasting can accumulate error over time.

## Models

The project includes several forecasting approaches:

- Naive baseline
- ARIMA
- Random Forest
- Exploratory LSTM implementation

The final frontend prototype focuses on the more stable forecasting options. LSTM was explored during development, but was not prioritised as a main user-facing model due to reliability and integration limitations.

## Frontend

The frontend prototype is located in the `frontend/` folder.

It was built using:

- React
- Vite
- JavaScript
- Recharts

The frontend allows users to:

- Select a supported stock ticker
- Select a forecasting model
- Select a forecast horizon
- Generate a forecast
- View results in a chart and table

To run the frontend:

```bash
cd frontend
npm install
npm run dev
```

The frontend usually runs at:

```text
http://localhost:5173
```

The backend must also be running for forecasts to work.

## Testing

Testing included:

- API testing through browser requests
- Valid ticker testing
- Invalid ticker testing
- Model selection testing
- Forecast horizon testing
- Frontend-to-backend integration testing
- Frontend error handling when the backend is unavailable
- Model evaluation using MAE and RMSE
- Walk-forward validation over recent trading data

## Current Status

Completed:

- Data retrieval and preprocessing pipeline
- Naive baseline model
- ARIMA forecasting
- Random Forest forecasting
- Exploratory LSTM integration
- FastAPI `/forecast` endpoint
- React frontend prototype
- Forecast visualisation using Recharts
- Model benchmarking and evaluation

Future improvements:

- Historical price overlays
- Confidence or uncertainty indicators
- More detailed model comparison views
- Wider ticker evaluation
- Deployment of backend and frontend
- Improved deep learning model integration

## Disclaimer

This project was developed for academic research purposes only. It does not constitute financial advice and should not be used for real-world trading or investment decisions.
