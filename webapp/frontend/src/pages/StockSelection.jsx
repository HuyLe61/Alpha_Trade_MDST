import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchStocks } from "../api";

const SECTOR_COLORS = {
  Technology: "border-blue-500/40 bg-blue-500/5",
  Financial: "border-amber-500/40 bg-amber-500/5",
  Healthcare: "border-emerald-500/40 bg-emerald-500/5",
  Energy: "border-orange-500/40 bg-orange-500/5",
  "Consumer Cyclical": "border-pink-500/40 bg-pink-500/5",
  "Consumer Defensive": "border-teal-500/40 bg-teal-500/5",
  Industrials: "border-slate-400/40 bg-slate-400/5",
  Communication: "border-purple-500/40 bg-purple-500/5",
};

const SECTOR_DOTS = {
  Technology: "bg-blue-500",
  Financial: "bg-amber-500",
  Healthcare: "bg-emerald-500",
  Energy: "bg-orange-500",
  "Consumer Cyclical": "bg-pink-500",
  "Consumer Defensive": "bg-teal-500",
  Industrials: "bg-slate-400",
  Communication: "bg-purple-500",
};

export default function StockSelection({ selected, onConfirm }) {
  const [stocks, setStocks] = useState([]);
  const [picked, setPicked] = useState(new Set(selected));
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchStocks()
      .then(setStocks)
      .catch(() => setStocks([]))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (ticker) => {
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker);
      else next.add(ticker);
      return next;
    });
  };

  const selectAll = () => setPicked(new Set(stocks.map((s) => s.ticker)));
  const clearAll = () => setPicked(new Set());

  const handleContinue = () => {
    onConfirm([...picked]);
    navigate("/configure");
  };

  const sectors = [...new Set(stocks.map((s) => s.sector))];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin h-8 w-8 border-2 border-brand-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold">Select Your Stock Universe</h1>
          <p className="text-gray-400 mt-1">
            Pick the stocks you want to trade. {picked.size} selected.
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={selectAll} className="btn-secondary text-xs">
            Select All
          </button>
          <button onClick={clearAll} className="btn-secondary text-xs">
            Clear
          </button>
          <button
            onClick={handleContinue}
            disabled={picked.size === 0}
            className="btn-primary"
          >
            Continue →
          </button>
        </div>
      </div>

      {/* Sector legend */}
      <div className="flex flex-wrap gap-3">
        {sectors.map((s) => (
          <span key={s} className="flex items-center gap-1.5 text-xs text-gray-400">
            <span className={`w-2 h-2 rounded-full ${SECTOR_DOTS[s] || "bg-gray-500"}`} />
            {s}
          </span>
        ))}
      </div>

      {/* Stock grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {stocks.map((s) => {
          const active = picked.has(s.ticker);
          return (
            <button
              key={s.ticker}
              onClick={() => toggle(s.ticker)}
              className={`relative rounded-xl border p-4 text-left transition-all duration-150 ${
                active
                  ? "border-brand-500 bg-brand-500/10 ring-1 ring-brand-500/30"
                  : `${SECTOR_COLORS[s.sector] || "border-gray-700 bg-gray-800/40"} hover:border-gray-600`
              }`}
            >
              {active && (
                <span className="absolute top-2 right-2 w-5 h-5 rounded-full bg-brand-500 flex items-center justify-center text-[10px] font-bold text-white">
                  ✓
                </span>
              )}
              <div className="text-lg font-bold">{s.ticker}</div>
              <div className="text-xs text-gray-400 mt-0.5 truncate">{s.name}</div>
              <div className="flex items-center gap-1.5 mt-2">
                <span className={`w-1.5 h-1.5 rounded-full ${SECTOR_DOTS[s.sector] || "bg-gray-500"}`} />
                <span className="text-[10px] text-gray-500">{s.sector}</span>
              </div>
              <div className="text-[10px] text-gray-600 mt-1">
                {s.min_date} — {s.max_date}
              </div>
            </button>
          );
        })}
      </div>

      {stocks.length === 0 && (
        <div className="card text-center py-16 text-gray-500">
          No stocks in database. Run <code className="text-brand-400">python fetch_data.py</code> first.
        </div>
      )}
    </div>
  );
}
