import sqlite3
import os
import pandas as pd
import numpy as np
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", str(Path(__file__).parent / "stocks.db")))

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "BAC", "GS",
    "JNJ", "UNH", "PFE",
    "XOM", "CVX",
    "PG", "KO", "WMT", "HD", "DIS",
    "BA", "CAT", "GE",
    "V", "MA",
    "NFLX", "CRM", "INTC", "AMD", "CSCO",
]

TICKER_INFO = {
    "AAPL": ("Apple", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "GOOGL": ("Alphabet", "Technology"),
    "AMZN": ("Amazon", "Consumer Cyclical"),
    "NVDA": ("NVIDIA", "Technology"),
    "META": ("Meta Platforms", "Technology"),
    "TSLA": ("Tesla", "Consumer Cyclical"),
    "JPM": ("JPMorgan Chase", "Financial"),
    "BAC": ("Bank of America", "Financial"),
    "GS": ("Goldman Sachs", "Financial"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "UNH": ("UnitedHealth", "Healthcare"),
    "PFE": ("Pfizer", "Healthcare"),
    "XOM": ("ExxonMobil", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "PG": ("Procter & Gamble", "Consumer Defensive"),
    "KO": ("Coca-Cola", "Consumer Defensive"),
    "WMT": ("Walmart", "Consumer Defensive"),
    "HD": ("Home Depot", "Consumer Cyclical"),
    "DIS": ("Walt Disney", "Communication"),
    "BA": ("Boeing", "Industrials"),
    "CAT": ("Caterpillar", "Industrials"),
    "GE": ("GE Aerospace", "Industrials"),
    "V": ("Visa", "Financial"),
    "MA": ("Mastercard", "Financial"),
    "NFLX": ("Netflix", "Communication"),
    "CRM": ("Salesforce", "Technology"),
    "INTC": ("Intel", "Technology"),
    "AMD": ("AMD", "Technology"),
    "CSCO": ("Cisco", "Technology"),
}


def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_prices(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_prices(date)")
    conn.commit()
    conn.close()


def insert_stock_data(ticker: str, df: pd.DataFrame):
    conn = get_connection()
    rows = []
    for date, row in df.iterrows():
        date_str = str(date)[:10]
        rows.append((
            ticker, date_str,
            float(row["Open"]), float(row["High"]),
            float(row["Low"]), float(row["Close"]),
            float(row["Volume"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO stock_prices (ticker, date, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def get_available_tickers():
    conn = get_connection()
    cur = conn.execute(
        "SELECT ticker, MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as n_rows "
        "FROM stock_prices GROUP BY ticker ORDER BY ticker"
    )
    results = []
    for row in cur.fetchall():
        ticker = row[0]
        name, sector = TICKER_INFO.get(ticker, (ticker, "Other"))
        results.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "min_date": row[1],
            "max_date": row[2],
            "n_rows": row[3],
        })
    conn.close()
    return results


def load_stock_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    conn = get_connection()
    stock_data = {}
    for ticker in tickers:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM stock_prices "
            "WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
            conn,
            params=(ticker, start_date, end_date),
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        stock_data[ticker] = df
    conn.close()
    return stock_data
