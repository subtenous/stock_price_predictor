import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const SUPPORTED_TICKERS = [
  "AAPL",
  "MSFT",
  "TSLA",
  "GOOGL",
  "AMZN",
  "^DJI",
  "^GSPC",
  "^IXIC",
];

const SUPPORTED_MODELS = [
  { value: "arima", label: "ARIMA" },
  { value: "rf", label: "Random Forest" },
  { value: "naive", label: "Naive Baseline" },
];

function App() {
  const [symbol, setSymbol] = useState("AAPL");
  const [model, setModel] = useState("arima");
  const [days, setDays] = useState(5);
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  

  const getPrediction = async () => {
    setLoading(true);
    setError("");
    setPredictionData(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/forecast?symbol=${encodeURIComponent(
          symbol.toUpperCase()
        )}&days=${days}&model=${model}`
      );

      if (!response.ok) {
        throw new Error("Forecast request failed");
      }

      const data = await response.json();
      setPredictionData(data);
    } catch (err) {
      setError(
        "Could not fetch forecast. Check the backend is running and the selected ticker/model is supported."
      );
    } finally {
      setLoading(false);
    }
  };

  const chartData =
    predictionData?.predictions?.map((item) => ({
      date: item.date,
      predictedClose: Number(item.predicted_close.toFixed(2)),
    })) || [];


  const modelLabel =
    SUPPORTED_MODELS.find((modelOption) => modelOption.value === model)?.label ||
    model.toUpperCase();

  return (
    <main className="app">
      <section className="card">
        <div className="header">
          <p className="tag">Final Year Project Prototype</p>
          <h1>Stock Price Predictor</h1>
          <p className="subtitle">
            Select a stock ticker and forecasting model to generate a short-term
            closing price prediction.
          </p>
        </div>

        <div className="controls">
          <div className="field">
            <label htmlFor="ticker">Ticker</label>
            <select
              id="ticker"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
            >
              {SUPPORTED_TICKERS.map((ticker) => (
                <option key={ticker} value={ticker}>
                  {ticker}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="model">Model</label>
            <select
              id="model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              {SUPPORTED_MODELS.map((modelOption) => (
                <option key={modelOption.value} value={modelOption.value}>
                  {modelOption.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="days">Forecast days</label>
            <select
              id="days"
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
            >
              <option value={1}>1 day</option>
              <option value={3}>3 days</option>
              <option value={5}>5 days</option>
              <option value={10}>10 days</option>
              <option value={15}>15 days</option>
              <option value={20}>20 days</option>
              <option value={30}>30 days</option>
            </select>
          </div>

          <button onClick={getPrediction} disabled={loading}>
            {loading ? "Predicting..." : "Generate Forecast"}
          </button>
        </div>

        <p className="supported-text">
          Supported tickers: {SUPPORTED_TICKERS.join(", ")}
        </p>

        {error && <p className="error">{error}</p>}

        {!predictionData && !loading && !error && (
          <div className="empty-state">
            <h2>No forecast loaded yet</h2>
            <p>
              Choose a ticker and model, then generate a forecast to view the
              predicted closing prices.
            </p>
          </div>
        )}

        {predictionData && (
          <div className="results">
            <div className="results-header">
              <div>
                <h2>{predictionData.symbol} Forecast</h2>
                <p>
                  Showing a {predictionData.horizon_days}-day forecast generated using the {modelLabel} model.
                </p>
              </div>
            </div>

            <div className="summary-grid">
              <div>
                <span>As of</span>
                <strong>{predictionData.as_of}</strong>
              </div>

              <div>
                <span>Forecast horizon</span>
                <strong>{predictionData.horizon_days} days</strong>
              </div>

              <div>
                <span>Selected model</span>
                <strong>{modelLabel}</strong>
              </div>

              <div>
                <span>Model order</span>
                <strong>
                  {predictionData.model_info?.order
                    ? `(${predictionData.model_info.order.join(", ")})`
                    : "N/A"}
                </strong>
              </div>

              <div>
                <span>Observations</span>
                <strong>{predictionData.model_info?.n_obs ?? "N/A"}</strong>
              </div>
            </div>

            <div className="chart-box">
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={["auto", "auto"]} />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="predictedClose"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                    activeDot={{ r: 7 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Predicted Close</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((item) => (
                  <tr key={item.date}>
                    <td>{item.date}</td>
                    <td>${item.predictedClose}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}

export default App;