# Stock Price Prediction System

This repository contains the **research and backend implementation** for my Final Year Project (University of Portsmouth).

The project focuses on **evaluating, improving, and deploying stock price forecasting models**, and exposing predictions via a REST API for use in a React frontend.

---

## Background

Financial markets are complex and noisy, making accurate forecasting challenging. While many machine learning and deep learning models have been proposed in academic literature, relatively few are evaluated rigorously and deployed in accessible applications.

This project explores **ensemble-based stock forecasting**, combining multiple time-series and machine learning models into a single meta-learner, and focuses on:
- model evaluation over realistic time horizons
- multi-day forecasting into the future
- deployment through an API suitable for frontend integration

---

## Attribution / Starting Point

This work was initially based on and adapted from the following open-source research repository:

- https://github.com/micfun123/AI-stock-prediction

The original repository provided a stacked-model architecture and initial experimentation framework.  
All subsequent modifications, extensions, evaluation, and deployment work are my own.

---

## My Contributions So Far

Key changes and additions implemented in this repository include:

- Refactored data range handling (customisable start/end dates, training windows)
- Introduced **recursive multi-step forecasting** (e.g. 30-day horizon) for LSTM-based models
- Implemented a **FastAPI backend** to expose forecasts via HTTP endpoints
- Added infrastructure for saving trained model artifacts and scalers
- Began restructuring the project for integration with a React frontend
- Improved clarity around training vs inference workflows

These changes move the project from a research-only prototype toward a **deployable system**.

---

## Model Architecture (Overview)

The system uses a **stacked ensemble approach**, where multiple base learners generate predictions that are combined by a meta-learner:


```
          Historical Prices ───► ARIMA ───┐
                                          │
           Time Series Data ───►  LSTM ───┤
                                          │
           Time Series Data ───►  GRU  ───┤
                                          │
  Technical Indicators & Features ─►  RF ─┤
                                          │
                                          ▼
                                Meta Learner: XGBoost
                                  (Final Prediction)
```

## Repository Structure

- learners/ # Individual base model training scripts (LSTM, GRU, ARIMA, RF, Transformer)
- meta_learner/ # XGBoost meta-learner training and evaluation
- utils/ # Data loading and preprocessing utilities
- start.py # Main orchestration script (data loading, training, evaluation)
- api_live.py # FastAPI backend exposing forecasting endpoints
- forecast_lstm_live.py # Recursive multi-step LSTM forecasting logic
- requirements.txt # Python dependencies
- README.md

---

## Data Source

Historical stock market data is retrieved programmatically using **Yahoo Finance** via the `yfinance` library.

The system supports configurable:
- ticker symbols
- training date ranges
- external comparison indices (e.g. market volatility or benchmark indices)

This allows experiments to be repeated across multiple assets and time periods.

---

## Forecasting Approach

The project supports both **single-step evaluation** and **multi-step recursive forecasting**.

For multi-day forecasting:
- the model predicts the next timestep
- the prediction is fed back as input
- the process is repeated for a fixed horizon (e.g. 30 days)

This approach reflects how forecasts would be generated in real-world usage, rather than relying on future ground-truth data.

---

## API Overview

A FastAPI backend is used to expose trained models via HTTP endpoints.  
This enables separation between:
- model training and research
- frontend visualisation and interaction

Example endpoint (development):
---

## Data Source

Historical stock market data is retrieved programmatically using **Yahoo Finance** via the `yfinance` library.

The system supports configurable:
- ticker symbols
- training date ranges
- external comparison indices (e.g. market volatility or benchmark indices)

This allows experiments to be repeated across multiple assets and time periods.

---

## Forecasting Approach

The project supports both **single-step evaluation** and **multi-step recursive forecasting**.

For multi-day forecasting:
- the model predicts the next timestep
- the prediction is fed back as input
- the process is repeated for a fixed horizon (e.g. 30 days)

This approach reflects how forecasts would be generated in real-world usage, rather than relying on future ground-truth data.

---

## API Overview

A FastAPI backend is used to expose trained models via HTTP endpoints.  
This enables separation between:
- model training and research
- frontend visualisation and interaction

Example endpoint (development):
GET /forecast?symbol=AAPL&days=30

Returns a JSON response containing:
- the forecast horizon
- the reference date
- predicted closing prices for future business days

The API layer is designed to be consumed by a React frontend.

---

## Current Status

This repository represents the **research and backend phase** of the project.

**Completed:**
- base learner training pipeline
- ensemble meta-learner
- recursive multi-day forecasting
- API integration for live inference

**In progress / upcoming:**
- multi-ticker robustness evaluation
- comparison across different training windows
- React frontend development
- formal evaluation and dissertation write-up

---

## Future Work

Planned extensions include:
- evaluating generalisation across a wider range of equities
- adding confidence or uncertainty indicators to forecasts
- improving model comparison and validation strategies
- frontend-based visualisation of predictions and historical trends

---

## Disclaimer

This project is conducted for **academic research purposes only**.  
It does **not** constitute financial advice and should not be used for real-world trading decisions.

