-- Alpha Trade live paper-trading database.
--
-- Writers: the trader worker (trader/), using the service-role key.
-- Readers: the public website (web/), using the anon key. Every table has RLS
-- on with a read-only policy, so the anon key can never write.

-- Market data the agent saw: 15-min bars for the 20 traded tickers, plus SPY
-- daily bars used as the benchmark on the equity chart.
create table if not exists public.market_bars (
  symbol     text        not null,
  timeframe  text        not null,          -- '15Min' | '1Day'
  ts         timestamptz not null,          -- bar start
  open       double precision not null,
  high       double precision not null,
  low        double precision not null,
  close      double precision not null,
  volume     double precision not null,
  primary key (symbol, timeframe, ts)
);
create index if not exists market_bars_ts_idx on public.market_bars (timeframe, ts desc);

-- Account equity over time. 'cycle' rows are written every 15 minutes by the
-- worker; 'history' rows are back-filled from Alpaca's portfolio history API.
create table if not exists public.account_snapshots (
  ts                timestamptz primary key,
  equity            double precision not null,
  cash              double precision,
  buying_power      double precision,
  long_market_value double precision,
  last_equity       double precision,       -- equity at previous market close
  source            text not null default 'cycle'
);

-- Current holdings, replaced on every sync.
create table if not exists public.positions (
  symbol           text primary key,
  qty              double precision not null,
  side             text,
  avg_entry_price  double precision,
  current_price    double precision,
  market_value     double precision,
  cost_basis       double precision,
  unrealized_pl    double precision,
  unrealized_plpc  double precision,
  updated_at       timestamptz not null default now()
);

-- One row per agent run: what it observed, what it decided, why it skipped.
create table if not exists public.agent_cycles (
  id               text primary key,        -- also embedded in each order's client_order_id
  ts               timestamptz not null default now(),
  status           text not null,           -- traded | no_orders | skipped | halted | error
  reason           text,
  model_version    text,
  portfolio_value  double precision,
  cash             double precision,
  buying_power     double precision,
  actions          jsonb,                   -- {"AAPL": 0.42, "MSFT": -0.10, ...} in [-1, 1]
  planned_orders   jsonb,                   -- [{symbol, side, qty, price}]
  n_orders         integer not null default 0,
  latency_ms       integer
);
create index if not exists agent_cycles_ts_idx on public.agent_cycles (ts desc);

-- Orders as Alpaca reports them (status, fills). Linked back to the cycle that placed them.
create table if not exists public.orders (
  id                uuid primary key,       -- Alpaca order id
  client_order_id   text,
  cycle_id          text,                   -- agent_cycles.id; null for orders placed by hand
  symbol            text not null,
  side              text not null,
  qty               double precision,
  filled_qty        double precision,
  filled_avg_price  double precision,
  order_type        text,
  status            text not null,
  submitted_at      timestamptz,
  filled_at         timestamptz,
  updated_at        timestamptz
);
create index if not exists orders_submitted_idx on public.orders (submitted_at desc);
create index if not exists orders_cycle_idx on public.orders (cycle_id);

-- Single-row control panel. Flip enabled=false in the Supabase table editor to
-- pause the agent; halted is set by the worker when max drawdown is breached.
create table if not exists public.agent_state (
  id              integer primary key default 1 check (id = 1),
  enabled         boolean not null default true,
  halted          boolean not null default false,
  halted_reason   text,
  initial_value   double precision,
  peak_value      double precision,
  trade_step      integer not null default 0,
  started_at      timestamptz,
  last_cycle_at   timestamptz,
  updated_at      timestamptz not null default now()
);
insert into public.agent_state (id) values (1) on conflict (id) do nothing;

-- Last snapshot of each New York trading day. PostgREST caps responses at
-- 1000 rows, so long-range charts read this instead of every 15-min snapshot.
create or replace view public.equity_daily
with (security_invoker = true) as
select distinct on ((ts at time zone 'America/New_York')::date)
  (ts at time zone 'America/New_York')::date as day,
  ts,
  equity,
  cash,
  long_market_value
from public.account_snapshots
order by (ts at time zone 'America/New_York')::date, ts desc;

create or replace view public.benchmark_daily
with (security_invoker = true) as
select symbol, (ts at time zone 'America/New_York')::date as day, close
from public.market_bars
where timeframe = '1Day';

-- Read-only public access for the website.
do $$
declare t text;
begin
  foreach t in array array['market_bars', 'account_snapshots', 'positions', 'agent_cycles', 'orders', 'agent_state']
  loop
    execute format('alter table public.%I enable row level security', t);
    execute format('drop policy if exists "public read" on public.%I', t);
    execute format('create policy "public read" on public.%I for select to anon, authenticated using (true)', t);
  end loop;
end $$;

-- Push new rows to the dashboard over Supabase Realtime.
do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime') then
    begin
      alter publication supabase_realtime add table public.account_snapshots, public.agent_cycles, public.orders;
    exception when duplicate_object then null;
    end;
  end if;
end $$;
