## Background
This repository is part of my Final Year Project (University of Portsmouth). The goal is to evaluate and improve a stock forecasting model and deploy it via an API and React frontend.

## Attribution / Starting Point
This work is based on and adapted from:
https://github.com/micfun123/AI-stock-prediction

## My Contributions So Far
- Modified data range configuration (e.g., start date / end date handling).
- Began implementing 30-day recursive forecasting for LSTM.
- Started a FastAPI backend to expose forecasting via HTTP endpoints.






In this project I have looked at stacked ML models as shown below to predict stocked prices. This was part of a wider university project.

`start.py` will download all required data from yahoo finance. `start.py` contains all config data as well such as the target ticker, External comparison tickers and file paths for saved predictions.

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
