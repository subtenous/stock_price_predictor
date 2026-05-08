# Frontend - Stock Price Prediction System

This folder contains the React frontend prototype for the Stock Price Prediction System.

The frontend was developed to demonstrate how users can interact with the FastAPI forecasting backend through a simple user-facing interface.

## Features

The frontend allows users to:

- Select a supported stock ticker
- Select a forecasting model
- Select a forecast horizon in trading days
- Request a forecast from the backend API
- View predicted closing prices in a line chart
- View exact forecast values in a table
- See loading and error messages

## Technologies Used

- React
- Vite
- JavaScript
- Recharts

## Setup

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The frontend should run locally at:

```text
http://localhost:5173
```

## Backend Requirement

The FastAPI backend must be running before forecasts can be generated.

From the main project folder, run:

```bash
uvicorn api_live:app --reload
```

The backend should run at:

```text
http://127.0.0.1:8000
```

## API Connection

The frontend sends forecast requests to the backend `/forecast` endpoint.

Example request:

```text
http://127.0.0.1:8000/forecast?symbol=AAPL&days=5&model=arima
```

The response is returned as JSON and displayed in the frontend as a chart and table.

## Forecast Controls

The frontend includes dropdown controls for:

- Stock ticker
- Forecasting model
- Forecast horizon

Dropdowns were used instead of unrestricted text input to reduce invalid requests and keep the prototype focused on supported options.

## Notes

The forecast horizon is measured in trading days rather than calendar days. This means weekends are skipped in the returned forecast dates.

Longer forecasts, such as 30 trading days, should be interpreted with caution because recursive forecasting can accumulate error over time.

## Prototype Scope

This frontend is a functional prototype designed to demonstrate the end-to-end workflow of the system. It is not intended to be a fully commercial trading platform.

Future improvements could include:

- Historical price overlays
- Model comparison views
- Confidence indicators
- Improved styling
- Deployment
- More detailed explanations of forecast outputs

## Disclaimer

This project is for academic purposes only. Forecasts should not be treated as financial advice.
