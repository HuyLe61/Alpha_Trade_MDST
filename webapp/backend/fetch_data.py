"""
Download 30 stocks x 15 years of OHLCV data from yfinance and store in SQLite.
Run once: python fetch_data.py
"""
import sys
import yfinance as yf
from database import TICKERS, init_db, insert_stock_data, get_connection

START = "2011-01-01"
END = "2026-01-01"


def fetch_and_store():
    init_db()
    failed = []

    for i, ticker in enumerate(TICKERS):
        print(f"[{i+1}/{len(TICKERS)}] Downloading {ticker} ...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if df.empty:
                print("EMPTY")
                failed.append(ticker)
                continue
            # yfinance may return MultiIndex columns for single ticker
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = df.columns.get_level_values(0)
            insert_stock_data(ticker, df)
            print(f"OK  ({len(df)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(ticker)

    conn = get_connection()
    cur = conn.execute("SELECT COUNT(DISTINCT ticker) FROM stock_prices")
    n_tickers = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM stock_prices")
    n_rows = cur.fetchone()[0]
    conn.close()

    print(f"\nDone. {n_tickers} tickers, {n_rows} total rows in DB.")
    if failed:
        print(f"Failed tickers: {failed}")


if __name__ == "__main__":
    fetch_and_store()
