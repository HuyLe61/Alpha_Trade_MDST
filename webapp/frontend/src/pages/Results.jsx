import { useNavigate } from "react-router-dom";
import MetricCard from "../components/MetricCard";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts";

const COLORS = ["#3388ff", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

export default function Results({ results }) {
  const navigate = useNavigate();

  if (!results) {
    return (
      <div className="card text-center py-16 text-gray-500">
        No results yet. <a href="/configure" className="text-brand-400 underline">Train a model</a>.
      </div>
    );
  }

  const { algorithm, training_stats, backtest } = results;
  const { metrics, portfolio_values, baseline_values, comparison } = backtest;
  const isMvp = algorithm === "mvp";

  // Build portfolio value chart data
  const valueData = portfolio_values.map((v, i) => {
    const point = { step: i, agent: +v.toFixed(2) };
    if (baseline_values) {
      Object.entries(baseline_values).forEach(([name, vals]) => {
        if (vals[i] != null) point[name] = +vals[i].toFixed(2);
      });
    }
    return point;
  });

  // Training stats charts
  const rewardData = (training_stats.iterations || []).map((iter, i) => ({
    iter,
    reward: +(training_stats.mean_rewards?.[i] ?? 0).toFixed(4),
  }));

  const lossData = (training_stats.iterations || []).map((iter, i) => ({
    iter,
    policy: +(training_stats.policy_losses?.[i] ?? 0).toFixed(5),
    value: +(training_stats.value_losses?.[i] ?? 0).toFixed(5),
  }));

  // Comparison table for MVP
  const comparisonRows = comparison
    ? Object.entries(comparison).map(([name, vals]) => ({
        name: name.replace(/_/g, " "),
        ...vals,
      }))
    : null;

  // Comparison bar chart
  const compBarData = comparisonRows
    ? comparisonRows.map((r) => ({
        name: r.name,
        "Total Return": +(r.return ?? 0).toFixed(2),
        "Sharpe Ratio": +(r.sharpe ?? 0).toFixed(3),
      }))
    : null;

  const baselineKeys = baseline_values ? Object.keys(baseline_values) : [];

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold">Backtest Results</h1>
          <p className="text-gray-400 mt-1">
            Algorithm: <span className="text-white font-semibold">{algorithm.toUpperCase()}</span>
          </p>
        </div>
        <button onClick={() => navigate("/")} className="btn-secondary">
          New Experiment
        </button>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Total Return"
          value={metrics.total_return.toFixed(2)}
          unit="%"
          positive={metrics.total_return > 0}
        />
        <MetricCard
          label="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(3)}
          positive={metrics.sharpe_ratio > 0}
        />
        <MetricCard
          label="Max Drawdown"
          value={metrics.max_drawdown.toFixed(2)}
          unit="%"
          positive={false}
        />
        <MetricCard
          label="Final Value"
          value={`$${metrics.final_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          positive={metrics.final_value > 100000}
        />
      </div>

      {/* Portfolio value chart */}
      <div className="card">
        <h2 className="text-sm font-semibold text-gray-300 mb-4">
          Portfolio Value Over Test Period
        </h2>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={valueData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#6b7280" }} label={{ value: "Trading Day", position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 11 }} />
            <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
              formatter={(v) => [`$${v.toLocaleString()}`, undefined]}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="agent" stroke="#3388ff" dot={false} strokeWidth={2} name="RL Agent" />
            {baselineKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[(i + 1) % COLORS.length]}
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="4 2"
                name={key.replace(/_/g, " ")}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Comparison table (MVP only) */}
      {comparisonRows && (
        <div className="card overflow-x-auto">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">
            Agent vs. Classical Baselines
          </h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Strategy</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Final Value</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Return %</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((r, i) => (
                <tr key={r.name} className={`border-b border-gray-800/50 ${i === 0 ? "bg-brand-500/5" : ""}`}>
                  <td className="py-2 px-3 font-medium capitalize">{r.name}</td>
                  <td className="py-2 px-3 text-right font-mono">
                    ${(r.value ?? 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                  </td>
                  <td className={`py-2 px-3 text-right font-mono ${(r.return ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}`}>
                    {(r.return ?? 0).toFixed(2)}%
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{(r.sharpe ?? 0).toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Comparison bar chart */}
      {compBarData && (
        <div className="card">
          <h2 className="text-sm font-semibold text-gray-300 mb-4">Return Comparison</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={compBarData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#6b7280" }} />
              <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
              <Tooltip
                contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
              />
              <Bar dataKey="Total Return" radius={[4, 4, 0, 0]}>
                {compBarData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Training stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Training Reward Curve</h3>
          {rewardData.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rewardData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="iter" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                />
                <Line type="monotone" dataKey="reward" stroke="#3388ff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">No data</div>
          )}
        </div>

        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Training Loss Curves</h3>
          {lossData.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="iter" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="policy" stroke="#f59e0b" dot={false} strokeWidth={2} name="Policy Loss" />
                <Line type="monotone" dataKey="value" stroke="#ef4444" dot={false} strokeWidth={2} name="Value Loss" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-gray-600 text-sm">No data</div>
          )}
        </div>
      </div>
    </div>
  );
}
