import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load env
if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

# ✅ Fetch DB URI
DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is missing from .env")

# ✅ Create DB engine
engine = create_engine(DATABASE_URL)

# ✅ Create stock_prices table if not exists
with engine.connect() as conn:
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
    print("✅ Table `analytics.stock_prices` ready.")

def fetch_and_store_stock(symbol="AAPL", table_name="stock_prices"):
    print(f"📥 Downloading {symbol} from Yahoo Finance...")
    df = yf.download(symbol, period="1d", interval="1d")
    df.reset_index(inplace=True)
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

    print("🔍 Sample data:")
    print(df.head())

    print(f"🗃 Writing to PostgreSQL → schema=analytics, table={table_name}")
    df.to_sql(
        table_name,
        engine,
        schema="analytics",
        if_exists="append",
        index=False
    )

    print("✅ Upload complete!")

# ✅ Run it
if __name__ == "__main__":
    fetch_and_store_stock()
