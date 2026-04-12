export default function MetricCard({ label, value, unit = "", positive }) {
  const color =
    positive === true
      ? "text-emerald-400"
      : positive === false
      ? "text-red-400"
      : "text-white";

  return (
    <div className="card flex flex-col gap-1">
      <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
        {label}
      </span>
      <span className={`text-2xl font-bold ${color}`}>
        {value}
        {unit && <span className="text-sm font-normal ml-1">{unit}</span>}
      </span>
    </div>
  );
}
