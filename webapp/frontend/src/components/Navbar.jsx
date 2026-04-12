import { Link, useLocation } from "react-router-dom";

const steps = [
  { path: "/", label: "1. Stocks" },
  { path: "/configure", label: "2. Configure" },
  { path: "/training", label: "3. Train" },
  { path: "/results", label: "4. Results" },
];

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-md sticky top-0 z-50">
      <div className="mx-auto max-w-7xl px-4 flex items-center h-14 gap-8">
        <Link to="/" className="text-lg font-bold tracking-tight text-white">
          AlgoTrade
        </Link>

        <div className="flex gap-1 ml-8">
          {steps.map((s) => {
            const active = pathname === s.path;
            return (
              <Link
                key={s.path}
                to={s.path}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition ${
                  active
                    ? "bg-brand-600/20 text-brand-400"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                {s.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
