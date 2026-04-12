import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { connectTrainingWs } from "../api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

export default function Training({ jobId, onComplete }) {
  const [status, setStatus] = useState("connecting");
  const [progress, setProgress] = useState(0);
  const [iteration, setIteration] = useState(0);
  const [total, setTotal] = useState(0);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [lossHistory, setLossHistory] = useState([]);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!jobId) return;

    const ws = connectTrainingWs(jobId, (data) => {
      if (data.status) setStatus(data.status);
      if (data.progress != null) setProgress(data.progress);
      if (data.current_iteration != null) setIteration(data.current_iteration);
      if (data.total_iterations != null) setTotal(data.total_iterations);
      if (data.error) setError(data.error);

      if (data.mean_reward != null) {
        setRewardHistory((prev) => [
          ...prev,
          { iter: data.current_iteration, reward: +data.mean_reward.toFixed(4) },
        ]);
      }
      if (data.policy_loss != null) {
        setLossHistory((prev) => [
          ...prev,
          {
            iter: data.current_iteration,
            policy: +data.policy_loss.toFixed(5),
            value: +data.value_loss.toFixed(5),
          },
        ]);
      }

      if (data.status === "completed" && data.results) {
        onComplete(data.results);
      }
    });

    wsRef.current = ws;
    return () => ws.close();
  }, [jobId]);

  const pct = Math.round(progress * 100);
  const isComplete = status === "completed";
  const isFailed = status === "failed";

  if (!jobId) {
    return (
      <div className="card text-center py-16 text-gray-500">
        No training job. <a href="/configure" className="text-brand-400 underline">Configure one</a>.
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-bold">
            {isComplete ? "Training Complete" : isFailed ? "Training Failed" : "Training in Progress"}
          </h1>
          <p className="text-gray-400 mt-1">
            Job <code className="text-brand-400">{jobId}</code>
            {total > 0 && ` — Iteration ${iteration} / ${total}`}
          </p>
        </div>
        {isComplete && (
          <button onClick={() => navigate("/results")} className="btn-primary">
            View Results →
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div className="card">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium capitalize">{status}</span>
          <span className="text-sm font-mono text-brand-400">{pct}%</span>
        </div>
        <div className="h-3 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-600 to-brand-400 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {error && (
        <div className="card border-red-500/40 bg-red-500/5 text-red-400">
          <span className="font-semibold">Error: </span>{error}
        </div>
      )}

      {/* Live charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Reward chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Mean Reward</h3>
          {rewardHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={rewardHistory}>
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
            <div className="flex items-center justify-center h-[220px] text-gray-600 text-sm">
              Waiting for data...
            </div>
          )}
        </div>

        {/* Loss chart */}
        <div className="card">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Losses</h3>
          {lossHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="iter" tick={{ fontSize: 10, fill: "#6b7280" }} />
                <YAxis tick={{ fontSize: 10, fill: "#6b7280" }} />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 12 }}
                />
                <Line type="monotone" dataKey="policy" stroke="#f59e0b" dot={false} strokeWidth={2} name="Policy Loss" />
                <Line type="monotone" dataKey="value" stroke="#ef4444" dot={false} strokeWidth={2} name="Value Loss" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[220px] text-gray-600 text-sm">
              Waiting for data...
            </div>
          )}
        </div>
      </div>

      {!isComplete && !isFailed && (
        <p className="text-xs text-gray-600 text-center">
          Charts update in real-time as training progresses.
        </p>
      )}
    </div>
  );
}
