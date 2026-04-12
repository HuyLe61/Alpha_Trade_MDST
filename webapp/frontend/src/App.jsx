import { useState } from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import StockSelection from "./pages/StockSelection";
import AlgoConfig from "./pages/AlgoConfig";
import Training from "./pages/Training";
import Results from "./pages/Results";

export default function App() {
  const [config, setConfig] = useState({
    tickers: [],
    algorithm: "",
    train_start: "2011-01-01",
    train_end: "2021-12-31",
    test_start: "2022-01-01",
    test_end: "2025-12-31",
    n_iterations: 100,
    steps_per_iter: 2048,
  });
  const [jobId, setJobId] = useState(null);
  const [results, setResults] = useState(null);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 mx-auto w-full max-w-7xl px-4 py-8">
        <Routes>
          <Route
            path="/"
            element={
              <StockSelection
                selected={config.tickers}
                onConfirm={(tickers) => setConfig((c) => ({ ...c, tickers }))}
              />
            }
          />
          <Route
            path="/configure"
            element={
              <AlgoConfig
                config={config}
                setConfig={setConfig}
                onJobStarted={setJobId}
              />
            }
          />
          <Route
            path="/training"
            element={
              <Training
                jobId={jobId}
                onComplete={(r) => setResults(r)}
              />
            }
          />
          <Route
            path="/results"
            element={<Results results={results} />}
          />
        </Routes>
      </main>
    </div>
  );
}
