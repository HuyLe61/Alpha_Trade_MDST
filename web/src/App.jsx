import { useEffect, useRef, useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import Overview from './pages/Overview.jsx'
import Dashboard from './pages/Dashboard.jsx'
import { DemoBanner } from './components/Live.jsx'
import { loadAll, subscribe } from './lib/data.js'

const REFRESH_MS = 60_000

function useLiveData() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const pending = useRef(null)

  useEffect(() => {
    let alive = true
    const load = () =>
      loadAll()
        .then((d) => alive && (setData(d), setError(null)))
        .catch((e) => alive && setError(e.message))
    // Realtime events arrive in bursts (one per row), so coalesce them.
    const soon = () => {
      clearTimeout(pending.current)
      pending.current = setTimeout(load, 1500)
    }
    load()
    const timer = setInterval(load, REFRESH_MS)
    const unsubscribe = subscribe(soon)
    return () => {
      alive = false
      clearInterval(timer)
      clearTimeout(pending.current)
      unsubscribe()
    }
  }, [])

  return { data, error }
}

export default function App() {
  const { data, error } = useLiveData()
  return (
    <>
      <DemoBanner />
      <header className="topbar">
        <NavLink to="/" className="wordmark">alpha<span>/</span>trade</NavLink>
        <nav className="nav">
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/dashboard">Dashboard</NavLink>
        </nav>
      </header>
      <main className="page">
        <Routes>
          <Route path="/" element={<Overview data={data} />} />
          <Route path="/dashboard" element={<Dashboard data={data} error={error} />} />
          <Route path="*" element={<p className="empty">There’s no page at this address. <NavLink to="/">Go to the overview</NavLink>.</p>} />
        </Routes>
      </main>
      <footer className="footer">
        <span>Alpha Trade</span>
        <span className="dim">Paper trading only · not investment advice</span>
      </footer>
    </>
  )
}
