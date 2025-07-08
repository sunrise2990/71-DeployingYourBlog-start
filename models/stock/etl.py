import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("Missing DB_URI in .env")

engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE SCHEMA IF NOT EXISTS analytics;
        CREATE TABLE IF NOT EXISTS analytics.stock_prices (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            date TIMESTAMP,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            volume BIGINT
        );
    """))
    print("✅ analytics.stock_prices ready")

def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"📥 Fetching data for: {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print(f"⚠️ No data returned for {symbol}")
        return

    # 1️⃣ Flatten MultiIndex column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # 2️⃣ Reset index and drop leftover name
    df.reset_index(inplace=True)
    df.columns.name = None

    # 3️⃣ Clean everything to strings (avoid tuples)
    df.columns = [str(col) for col in df.columns]

    # 4️⃣ Add symbol and rename
    df["symbol"] = symbol
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    print(df.head())
    df.to_sql(table_name, con=engine, schema="analytics",
              if_exists="append", index=False)
    print("✅ Uploaded")

if __name__ == "__main__":
    load_stock_data("AAPL")

