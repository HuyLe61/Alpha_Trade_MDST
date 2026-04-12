import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchAlgorithms, startTraining } from "../api";

const ENV_LABELS = {
  portfolio: "Portfolio (weight allocation, Sharpe reward, baselines)",
  naive: "Trade (buy/sell shares, P&L reward)",
};

export default function AlgoConfig({ config, setConfig, onJobStarted }) {
  const [algorithms, setAlgorithms] = useState([]);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchAlgorithms().then(setAlgorithms).catch(() => {});
  }, []);

  const selectedAlgo = algorithms.find((a) => a.id === config.algorithm);

  const handleTrain = async () => {
    if (!config.algorithm || config.tickers.length === 0) return;
    setSubmitting(true);
    try {
      const res = await startTraining(config);
      onJobStarted(res.job_id);
      navigate("/training");
    } catch (e) {
      alert("Failed to start training: " + e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold">Configure Training</h1>
        <p className="text-gray-400 mt-1">
          {config.tickers.length} stocks selected. Choose an algorithm and time ranges.
        </p>
      </div>

      {/* Selected stocks preview */}
      <div className="card">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Selected Stocks</h2>
        <div className="flex flex-wrap gap-2">
          {config.tickers.map((t) => (
            <span
              key={t}
              className="px-2.5 py-1 rounded-md bg-gray-800 border border-gray-700 text-xs font-medium"
            >
              {t}
            </span>
          ))}
          {config.tickers.length === 0 && (
            <span className="text-sm text-gray-500">
              No stocks selected.{" "}
              <a href="/" className="text-brand-400 underline">Go back</a>
            </span>
          )}
        </div>
      </div>

      {/* Algorithm picker */}
      <div className="card space-y-4">
        <h2 className="text-sm font-semibold text-gray-300">Algorithm</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {algorithms.map((algo) => {
            const active = config.algorithm === algo.id;
            return (
              <button
                key={algo.id}
                onClick={() => setConfig((c) => ({ ...c, algorithm: algo.id }))}
                className={`rounded-lg border p-4 text-left transition ${
                  active
                    ? "border-brand-500 bg-brand-500/10 ring-1 ring-brand-500/30"
                    : "border-gray-700 bg-gray-800/40 hover:border-gray-600"
                }`}
              >
                <div className="font-semibold text-sm">{algo.name}</div>
                <div className="text-xs text-gray-400 mt-1 line-clamp-2">{algo.description}</div>
                <div className="mt-2 text-[10px] px-2 py-0.5 rounded-full bg-gray-800 border border-gray-700 inline-block text-gray-400">
                  {ENV_LABELS[algo.env_type]}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Date ranges */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Training Period</h2>
          <div className="flex gap-3">
            <label className="flex-1">
              <span className="text-xs text-gray-500">Start</span>
              <input
                type="date"
                value={config.train_start}
                onChange={(e) => setConfig((c) => ({ ...c, train_start: e.target.value }))}
                className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
              />
            </label>
            <label className="flex-1">
              <span className="text-xs text-gray-500">End</span>
              <input
                type="date"
                value={config.train_end}
                onChange={(e) => setConfig((c) => ({ ...c, train_end: e.target.value }))}
                className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
              />
            </label>
          </div>
        </div>

        <div className="card space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Test Period</h2>
          <div className="flex gap-3">
            <label className="flex-1">
              <span className="text-xs text-gray-500">Start</span>
              <input
                type="date"
                value={config.test_start}
                onChange={(e) => setConfig((c) => ({ ...c, test_start: e.target.value }))}
                className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
              />
            </label>
            <label className="flex-1">
              <span className="text-xs text-gray-500">End</span>
              <input
                type="date"
                value={config.test_end}
                onChange={(e) => setConfig((c) => ({ ...c, test_end: e.target.value }))}
                className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
              />
            </label>
          </div>
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="card space-y-3">
        <h2 className="text-sm font-semibold text-gray-300">Hyperparameters</h2>
        <div className="grid grid-cols-2 gap-4">
          <label>
            <span className="text-xs text-gray-500">Iterations</span>
            <input
              type="number"
              min={1}
              max={1000}
              value={config.n_iterations}
              onChange={(e) =>
                setConfig((c) => ({ ...c, n_iterations: parseInt(e.target.value) || 50 }))
              }
              className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
            />
          </label>
          <label>
            <span className="text-xs text-gray-500">Steps per iteration</span>
            <input
              type="number"
              min={128}
              step={128}
              value={config.steps_per_iter}
              onChange={(e) =>
                setConfig((c) => ({ ...c, steps_per_iter: parseInt(e.target.value) || 2048 }))
              }
              className="mt-1 w-full rounded-md border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-gray-200 focus:border-brand-500 focus:outline-none"
            />
          </label>
        </div>
      </div>

      {/* Train button */}
      <div className="flex gap-3">
        <button onClick={() => navigate("/")} className="btn-secondary">
          ← Back
        </button>
        <button
          onClick={handleTrain}
          disabled={submitting || !config.algorithm || config.tickers.length === 0}
          className="btn-primary flex-1 max-w-xs"
        >
          {submitting ? "Starting..." : "Start Training"}
        </button>
      </div>
    </div>
  );
}
