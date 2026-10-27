// Everything the site reads. One shape, two sources: the live Supabase
// tables written by trader/, or generated demo data when no keys are set.
import { createClient } from '@supabase/supabase-js'
import { demoData } from './demo.js'

export const REPO = 'https://github.com/HuyLe61/Alpha_Trade_MDST'

const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = url && anonKey ? createClient(url, anonKey) : null
export const isDemo = !supabase

const INTRADAY_DAYS = 7

function unwrap({ data, error }, what) {
  if (error) throw new Error(`Couldn't load ${what}: ${error.message}`)
  return data
}

export async function loadAll() {
  if (isDemo) return demoData()

  const since = new Date(Date.now() - INTRADAY_DAYS * 864e5).toISOString()
  const [intraday, daily, benchmark, cycles, orders, positions, state, filled] = await Promise.all([
    supabase.from('account_snapshots').select('ts,equity,last_equity').gte('ts', since).order('ts'),
    supabase.from('equity_daily').select('day,ts,equity').order('day'),
    supabase.from('benchmark_daily').select('day,close').eq('symbol', 'SPY').order('day'),
    supabase.from('agent_cycles').select('*').order('ts', { ascending: false }).limit(80),
    supabase.from('orders').select('*').order('submitted_at', { ascending: false }).limit(60),
    supabase.from('positions').select('*').order('market_value', { ascending: false }),
    supabase.from('agent_state').select('*').eq('id', 1).maybeSingle(),
    supabase.from('orders').select('id', { count: 'exact', head: true }).eq('status', 'filled'),
  ])

  return {
    intraday: unwrap(intraday, 'intraday equity'),
    daily: unwrap(daily, 'daily equity'),
    benchmark: unwrap(benchmark, 'SPY benchmark'),
    cycles: unwrap(cycles, 'agent cycles'),
    orders: unwrap(orders, 'orders'),
    positions: unwrap(positions, 'positions'),
    state: unwrap(state, 'agent state'),
    filledCount: filled.count ?? 0,
    loadedAt: new Date(),
  }
}

// Calls onChange whenever the worker writes a new snapshot, cycle, or order.
export function subscribe(onChange) {
  if (isDemo) return () => {}
  const channel = supabase
    .channel('alpha-trade-live')
    .on('postgres_changes', { event: '*', schema: 'public', table: 'account_snapshots' }, onChange)
    .on('postgres_changes', { event: '*', schema: 'public', table: 'agent_cycles' }, onChange)
    .on('postgres_changes', { event: '*', schema: 'public', table: 'orders' }, onChange)
    .subscribe()
  return () => supabase.removeChannel(channel)
}
